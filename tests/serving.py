"""
Model serving for the prototype (Section 4.5.2 of the thesis).

Loads the artefact written by fraud_lr_pipeline.py (pipeline + threshold + version + feature
list) and scores ONE transaction. The decision is computed in one place, decide(), from the
same score and the same stored threshold that are returned to the caller.

Use it from the FastAPI route like this:

    from serving import load_artefact, score_transaction
    ART = load_artefact("models/fraud_lr_v1.joblib")      # once, at start-up

    @app.post("/predict")
    def predict(txn: TxnIn, user=Depends(current_user)):
        return score_transaction(ART, txn.dict())          # raises ValueError -> return 422
"""
import math

import joblib
import numpy as np
import pandas as pd

TYPES = {"CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"}

READABLE = {
    "log_amount": "transaction amount",
    "log_old_org": "sender's balance before the transaction",
    "log_old_dest": "recipient's balance before the transaction",
    "balance_ratio": "share of the sender's balance being moved",
    "hour": "hour of day",
    "night": "night-time (00:00 to 04:59)",
    "sweep": "transaction empties the sender's balance",
    "dest_merchant": "recipient is a merchant account",
    "dest_empty": "recipient account was empty",
}


def load_artefact(path):
    art = joblib.load(path)
    for k in ("pipeline", "threshold", "version", "features"):
        if k not in art:
            raise ValueError(f"artefact is missing '{k}'")
    return art


def validate(txn):
    """Reject bad input before it reaches the model (tests T11 and T12)."""
    need = ["type", "amount", "hour", "old_balance_org", "old_balance_dest", "name_dest"]
    missing = [k for k in need if k not in txn or txn[k] is None]
    if missing:
        raise ValueError(f"missing field(s): {', '.join(missing)}")
    if txn["type"] not in TYPES:
        raise ValueError(f"type must be one of {sorted(TYPES)}")
    for k in ("amount", "old_balance_org", "old_balance_dest"):
        v = float(txn[k])
        if not math.isfinite(v) or v < 0:
            raise ValueError(f"{k} must be a non-negative number")
    if not (0 <= int(txn["hour"]) <= 23):
        raise ValueError("hour must be between 0 and 23")


def make_features(txn):
    """One-row DataFrame built exactly as prepare() builds the training features."""
    validate(txn)
    amount = float(txn["amount"])
    old_org = max(float(txn["old_balance_org"]), 0.0)
    old_dest = max(float(txn["old_balance_dest"]), 0.0)
    hour = float(int(txn["hour"]))
    row = {
        "log_amount": np.log1p(amount),
        "log_old_org": np.log1p(old_org),
        "log_old_dest": np.log1p(old_dest),
        "balance_ratio": float(np.clip(amount / max(old_org, 1.0), 0, 1)),
        "hour": hour,
        "night": float(hour <= 4),
        "sweep": float(old_org > 0 and amount >= old_org),
        "dest_merchant": float(str(txn["name_dest"]).startswith("M")),
        "dest_empty": float(old_dest == 0),
        "type": str(txn["type"]),
    }
    return pd.DataFrame([row])


def decide(score, threshold):
    """THE decision rule. Nothing else in the system may decide fraud."""
    return bool(score >= threshold)


def _reasons(pipe, X, top=3):
    pre, clf = pipe.named_steps["pre"], pipe.named_steps["clf"]
    vals = pre.transform(X)[0]
    names = pre.get_feature_names_out()
    contrib = clf.coef_[0] * vals
    out = []
    for i in np.argsort(-contrib)[: top + 3]:
        if contrib[i] <= 0:
            break
        base = names[i].split("__", 1)[-1]
        if base.startswith("type_"):
            label = "transaction type " + base[5:]
        else:
            label = READABLE.get(base, base)
        out.append(f"{label} (+{contrib[i]:.2f})")
        if len(out) == top:
            break
    return out


def score_transaction(art, txn):
    X = make_features(txn)[art["features"]]
    score = float(art["pipeline"].predict_proba(X)[:, 1][0])
    thr = float(art["threshold"])
    return {
        "txn_id": txn.get("txn_id"),
        "is_fraud": decide(score, thr),
        "score": score,
        "threshold": thr,
        "reason": _reasons(art["pipeline"], X),
        "model_version": art["version"],
    }
