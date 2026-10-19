"""Integration tests for the prototype API (T1-T6, T9, T13, T16).
Run:  API_URL=http://localhost:8000 API_USER=analyst API_PASS=secret pytest -v test_api.py
Edit the ENDPOINT/field names below if your backend differs from Appendix B.
"""
import os, pytest, requests
API = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
USER = os.getenv("API_USER", "testuser"); PASS = os.getenv("API_PASS", "Test1234!")
REGISTER = os.getenv("REGISTER_PATH", "/auth/register")
LOGIN = os.getenv("LOGIN_PATH", "/auth/login"); PREDICT = os.getenv("PREDICT_PATH", "/predict")
LIST = os.getenv("LIST_PATH", "/transactions"); ADMIN = os.getenv("ADMIN_PATH", "/admin/users")
TOKEN_KEY = os.getenv("TOKEN_KEY", "access_token")

def _up():
    try: requests.get(API, timeout=5); return True
    except Exception: return False
pytestmark = pytest.mark.skipif(not _up(), reason="API not reachable at " + API)

TXN = {"txn_id": "T-TEST-1", "type": "TRANSFER", "amount": 181000.0, "hour": 3,
       "old_balance_org": 181000.0, "old_balance_dest": 0.0, "name_dest": "C123"}
def login(u=USER, p=PASS): return requests.post(API + LOGIN, json={"username": u, "password": p}, timeout=10)
def hdr():
    r = login()
    if r.status_code != 200:   # first run: create the test user, then log in
        requests.post(API + REGISTER, json={"username": USER, "email": USER + "@example.com", "password": PASS}, timeout=10)
        r = login()
    return {"Authorization": "Bearer " + r.json()[TOKEN_KEY]}

def test_T1_login_valid():
    hdr(); r = login(); assert r.status_code == 200 and TOKEN_KEY in r.json()
def test_T2_login_wrong_password():
    assert login(p="wrong-password").status_code in (400, 401, 403)
def test_T3_login_unknown_user():
    assert login(u="nobody-xyz").status_code in (400, 401, 403)
def test_T4_predict_valid():
    r = requests.post(API + PREDICT, json=TXN, headers=hdr(), timeout=10)
    assert r.status_code == 200
    b = r.json()
    for k in ("is_fraud", "score"): assert k in b
def test_T5_decision_matches_threshold():
    b = requests.post(API + PREDICT, json=TXN, headers=hdr(), timeout=10).json()
    if "threshold" in b: assert b["is_fraud"] == (b["score"] >= b["threshold"])
def test_T6_alert_listed():
    requests.post(API + PREDICT, json=TXN, headers=hdr(), timeout=10)
    r = requests.get(API + LIST, headers=hdr(), timeout=10); assert r.status_code == 200
def test_T9_predict_persisted():
    r = requests.get(API + LIST, headers=hdr(), timeout=10); assert r.status_code == 200 and len(r.json()) >= 1
def test_T13_no_or_bad_token_rejected():
    assert requests.post(API + PREDICT, json=TXN, timeout=10).status_code in (401, 403)
    assert requests.post(API + PREDICT, json=TXN, headers={"Authorization": "Bearer abc.def.ghi"}, timeout=10).status_code in (401, 403)
@pytest.mark.skipif(not ADMIN, reason="set ADMIN_PATH to test role restriction")
def test_T16_analyst_blocked_from_admin():
    assert requests.get(API + ADMIN, headers=hdr(), timeout=10).status_code in (401, 403)
