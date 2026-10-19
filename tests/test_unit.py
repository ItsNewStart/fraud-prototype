"""
Unit tests for Chapter Five, Table 5.2 (tests T7, T8, T11, T12, T14, T15).
Run:  pytest -v test_unit.py
Set these environment variables first:
    ARTEFACT=results/fraud_lr_v1.joblib
    PAYSIM_CSV=PS_20174392719_1491204439457_log.csv      (needed for T14 only)
"""
import os

import numpy as np
import pandas as pd
import pytest

import serving

ART_PATH = os.environ.get("ARTEFACT", "results/fraud_lr_v1.joblib")
CSV = os.environ.get("PAYSIM_CSV")


@pytest.fixture(scope="module")
def art():
    return serving.load_artefact(ART_PATH)


GOOD = dict(txn_id="t1", type="TRANSFER", amount=181.0, hour=3, old_balance_org=181.0, old_balance_dest=0.0, name_dest="C123")


def test_T7_T8_decision_consistency(art):
    """is_fraud must equal (score >= threshold) for scores on both sides of the threshold."""
    thr = art["threshold"]
    grid = np.concatenate([np.linspace(0, 1, 2001), [thr, thr - 1e-12, thr + 1e-12]])
    n_hi = n_lo = 0
    for s in grid:
        d = serving.decide(s, thr)
        assert d == (s >= thr)
        n_hi += d
        n_lo += (not d)
    print(f"\nT7: {n_hi} scores at/above threshold all true;  T8: {n_lo} scores below all false;  n = {len(grid)}")
    assert n_hi > 0 and n_lo > 0


def test_response_is_self_consistent(art):
    out = serving.score_transaction(art, GOOD)
    assert out["is_fraud"] == (out["score"] >= out["threshold"])
    assert set(out) >= {"is_fraud", "score", "threshold", "reason", "model_version"}


@pytest.mark.parametrize("field,value", [("amount", -5), ("old_balance_org", -1), ("hour", 24), ("hour", -1)])
def test_T11_rejects_bad_numbers(art, field, value):
    with pytest.raises(ValueError):
        serving.score_transaction(art, {**GOOD, field: value})


def test_T12_rejects_unknown_type(art):
    with pytest.raises(ValueError):
        serving.score_transaction(art, {**GOOD, "type": "WIRE"})


def test_missing_field_rejected(art):
    bad = dict(GOOD); bad.pop("amount")
    with pytest.raises(ValueError):
        serving.score_transaction(art, bad)


def test_T15_artefact_contents(art):
    for k in ("pipeline", "threshold", "version", "features"):
        assert k in art
    assert 0 < art["threshold"] < 1


@pytest.mark.skipif(not CSV, reason="set PAYSIM_CSV to run the feature-parity test")
def test_T14_feature_parity(art):
    """Serving features must equal the training-pipeline features for the same records."""
    import fraud_lr_pipeline as P

    raw = pd.read_csv(CSV, nrows=200000, usecols=P.COLS)
    d = P.prepare(raw, False)
    idx = np.random.default_rng(0).choice(len(raw), 300, replace=False)
    num = ["log_amount", "log_old_org", "log_old_dest", "balance_ratio", "hour", "night", "sweep", "dest_merchant", "dest_empty"]
    for i in idx:
        r = raw.iloc[i]
        f = serving.make_features(dict(type=r.type, amount=r.amount, hour=int(r.step % 24),
                                       old_balance_org=r.oldbalanceOrg, old_balance_dest=r.oldbalanceDest, name_dest=r.nameDest))
        assert np.allclose(f[num].values[0], d.iloc[i][num].astype(float).values), f"row {i}"
        assert f["type"].iloc[0] == d.iloc[i]["type"]
    print(f"\nT14: {len(idx)} records compared, all identical")
