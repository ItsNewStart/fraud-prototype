"""Fraud detection prototype: FastAPI backend (Sections 3.8 to 3.13, 4.5 and Appendix B).

Run:  uvicorn main:app --reload --port 8000        (from the backend folder)
Settings come from environment variables (see ../README.md); nothing secret is in the code.
Every prediction goes through serving.score_transaction(), so the decision is always
is_fraud = (score >= threshold), with the threshold stored in the model artefact.
"""
import csv, io, os, re, uuid
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from bson import ObjectId
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.responses import JSONResponse

from serving import load_artefact, score_transaction

try:                                   # reads backend/.env if present (settings for local runs)
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SECRET = os.getenv("JWT_SECRET", "change-me-in-production")
TOKEN_HOURS = int(os.getenv("TOKEN_HOURS", "8"))
ARTEFACT_PATH = os.getenv("ARTEFACT") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "fraud_lr_v1.joblib")
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", "2000"))
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

# ------------------------------------------------------------------ database
if os.getenv("USE_MOCK_DB") == "1":                      # for local testing only
    import mongomock
    DB = mongomock.MongoClient()["fraud_prototype"]
else:
    from pymongo import MongoClient
    DB = MongoClient(os.environ["MONGO_URI"])[os.getenv("DB_NAME", "fraud_prototype")]
USERS, TXNS, LOGS = DB["users"], DB["transactions"], DB["logs"]
USERS.create_index("username", unique=True)
TXNS.create_index("user_id"); TXNS.create_index("created_at"); LOGS.create_index("user_id")

ART = load_artefact(ARTEFACT_PATH)                        # model + preprocessing + threshold + version

def now(): return datetime.now(timezone.utc)

def log(user_id, action, status="ok", details=""):
    LOGS.insert_one({"user_id": str(user_id) if user_id else None, "action": action, "status": status, "details": str(details)[:500], "timestamp": now()})

def seed_admin():
    u, p = os.getenv("ADMIN_USERNAME"), os.getenv("ADMIN_PASSWORD")
    if u and p and not USERS.find_one({"username": u}):
        USERS.insert_one({"username": u, "email": os.getenv("ADMIN_EMAIL", "admin@example.com"),
                          "password_hash": bcrypt.hashpw(p.encode(), bcrypt.gensalt()).decode(),
                          "role": "admin", "created_at": now()})
seed_admin()

app = FastAPI(title="Payment fraud detection prototype")

@app.exception_handler(Exception)
async def unexpected(request: Request, exc: Exception):   # 500s are written to the Logs collection
    log(None, "server_error", "error", f"{request.url.path}: {exc}")
    return JSONResponse(status_code=500, content={"detail": "server error"})

# ------------------------------------------------------------------ auth
def current_user(request: Request):
    h = request.headers.get("authorization", "")
    if not h.lower().startswith("bearer "):
        raise HTTPException(401, "missing token")
    try:
        p = jwt.decode(h[7:], SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(401, "invalid or expired token")
    return {"user_id": p["uid"], "username": p["sub"], "role": p["role"]}

def admin_only(u=Depends(current_user)):
    if u["role"] != "admin":
        raise HTTPException(403, "administrator role required")
    return u

@app.post("/auth/register", status_code=201)
def register(body: dict):
    username, email, pw = (body.get("username") or "").strip(), (body.get("email") or "").strip(), body.get("password") or ""
    if len(username) < 3 or not EMAIL_RE.match(email) or len(pw) < 8:
        raise HTTPException(422, "username (3+ chars), a valid email and a password of 8+ characters are required")
    if USERS.find_one({"username": username}):
        raise HTTPException(409, "username already taken")
    r = USERS.insert_one({"username": username, "email": email, "role": "user", "created_at": now(),
                          "password_hash": bcrypt.hashpw(pw.encode(), bcrypt.gensalt()).decode()})
    log(r.inserted_id, "register")
    return {"message": "registered", "username": username}

@app.post("/auth/login")
def login(body: dict):
    u = USERS.find_one({"username": body.get("username")})
    ok = bool(u) and bcrypt.checkpw((body.get("password") or "").encode(), u["password_hash"].encode())
    log(u["_id"] if u else None, "login", "ok" if ok else "failed", body.get("username"))
    if not ok:
        raise HTTPException(401, "invalid username or password")
    tok = jwt.encode({"sub": u["username"], "uid": str(u["_id"]), "role": u["role"],
                      "exp": now() + timedelta(hours=TOKEN_HOURS)}, SECRET, "HS256")
    return {"access_token": tok, "token_type": "bearer", "role": u["role"]}

# ------------------------------------------------------------------ model
@app.get("/")
def health(): return {"status": "ok"}

@app.get("/model")
def model_info(u=Depends(current_user)):
    return {"model_version": ART["version"], "threshold": ART["threshold"]}

def _store(user, result, txn):
    doc = {"txn_id": result["txn_id"], "user_id": user["user_id"], "username": user["username"],
           "type": txn["type"], "amount": float(txn["amount"]), "hour": int(txn["hour"]),
           "old_balance_org": float(txn["old_balance_org"]), "old_balance_dest": float(txn["old_balance_dest"]),
           "name_dest": str(txn["name_dest"]), "is_fraud": result["is_fraud"], "score": result["score"],
           "threshold": result["threshold"], "reason": result["reason"], "model_version": result["model_version"],
           "created_at": now()}
    TXNS.insert_one(doc)

def _score(user, txn):
    txn = dict(txn); txn.setdefault("txn_id", uuid.uuid4().hex[:12])
    result = score_transaction(ART, txn)                  # ValueError on bad input
    if isinstance(result.get("reason"), list):            # Table 3.8: reason is a String
        result["reason"] = "; ".join(result["reason"])
    _store(user, result, txn)
    return result

@app.post("/predict")
def predict(txn: dict, u=Depends(current_user)):
    try:
        r = _score(u, txn)
    except (ValueError, TypeError) as e:
        log(u["user_id"], "predict", "rejected", e)
        raise HTTPException(422, str(e))
    log(u["user_id"], "predict", "ok", f"{r['txn_id']} fraud={r['is_fraud']}")
    return r

@app.post("/upload")
async def upload(file: UploadFile = File(...), u=Depends(current_user)):
    """Batch scoring (FR-03): CSV with columns type, amount, hour, old_balance_org, old_balance_dest, name_dest [, txn_id]."""
    text = (await file.read()).decode("utf-8-sig", errors="replace")
    rows = list(csv.DictReader(io.StringIO(text)))
    if not rows:
        raise HTTPException(422, "the CSV file is empty")
    if len(rows) > MAX_UPLOAD_ROWS:
        raise HTTPException(422, f"too many rows ({len(rows)}); the limit is {MAX_UPLOAD_ROWS}")
    results, rejected = [], []
    for i, row in enumerate(rows, start=2):
        try:
            results.append(_score(u, {k.strip(): v for k, v in row.items()}))
        except (ValueError, TypeError) as e:
            rejected.append({"line": i, "error": str(e)})
    log(u["user_id"], "upload", "ok", f"{file.filename}: {len(results)} scored, {len(rejected)} rejected")
    return {"scored": len(results), "flagged": sum(r["is_fraud"] for r in results), "rejected": rejected, "results": results}

# ------------------------------------------------------------------ transactions and dashboards
def _scope(u): return {} if u["role"] == "admin" else {"user_id": u["user_id"]}

def _clean(d):
    d = dict(d); d["id"] = str(d.pop("_id"))
    for k in ("created_at", "timestamp"):
        if k in d and hasattr(d[k], "isoformat"): d[k] = d[k].isoformat()
    return d

@app.get("/transactions")
def transactions(type: str = None, is_fraud: bool = None, date_from: str = Query(None, alias="from"),
                 date_to: str = Query(None, alias="to"), page: int = 1, page_size: int = 20, u=Depends(current_user)):
    q = _scope(u)
    if type: q["type"] = type
    if is_fraud is not None: q["is_fraud"] = is_fraud
    rng = {}
    try:
        if date_from: rng["$gte"] = datetime.fromisoformat(date_from).replace(tzinfo=timezone.utc)
        if date_to: rng["$lt"] = datetime.fromisoformat(date_to).replace(tzinfo=timezone.utc) + timedelta(days=1)
    except ValueError:
        raise HTTPException(422, "dates must look like 2026-10-13")
    if rng: q["created_at"] = rng
    page, page_size = max(page, 1), min(max(page_size, 1), 100)
    cur = TXNS.find(q).sort("created_at", -1).skip((page - 1) * page_size).limit(page_size)
    return {"page": page, "total": TXNS.count_documents(q), "items": [_clean(d) for d in cur]}

@app.get("/stats")
def stats(days: int = 14, u=Depends(current_user)):
    q = _scope(u); total = TXNS.count_documents(q); fraud = TXNS.count_documents({**q, "is_fraud": True})
    since = now() - timedelta(days=max(1, min(days, 365))); trend = {}
    for d in TXNS.find({**q, "created_at": {"$gte": since}}, {"created_at": 1, "is_fraud": 1}):
        k = d["created_at"].strftime("%Y-%m-%d"); t = trend.setdefault(k, {"date": k, "transactions": 0, "flagged": 0})
        t["transactions"] += 1; t["flagged"] += int(d["is_fraud"])
    return {"total": total, "flagged": fraud, "fraud_share": (fraud / total) if total else 0.0,
            "trend": sorted(trend.values(), key=lambda x: x["date"]),
            "model_version": ART["version"], "threshold": ART["threshold"]}

# ------------------------------------------------------------------ administration (FR-07)
@app.get("/admin/users")
def admin_users(u=Depends(admin_only)):
    out = []
    for d in USERS.find({}, {"password_hash": 0}):
        d = _clean(d); d["transactions"] = TXNS.count_documents({"user_id": d["id"]}); out.append(d)
    return out

@app.get("/admin/logs")
def admin_logs(limit: int = 100, u=Depends(admin_only)):
    return [_clean(d) for d in LOGS.find().sort("timestamp", -1).limit(min(limit, 500))]
