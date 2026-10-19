#!/usr/bin/env python3
"""
Logistic regression fraud detection pipeline and rule-based baseline on PaySim.
Reproduces the experiments in Chapters Three to Five of the thesis:
"Logistic Regression-Based Detection of Digital Payment Fraud in Nigeria:
A Comparison with Rule-Based Methods".

Usage:
    python fraud_lr_pipeline.py --data PS_log.csv --out results

Input: the PaySim CSV (Kaggle, 'ealaxi/paysim1') with the 11 original columns:
    step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
    nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

Design points (all explained in Chapter Three):
  * Only information available BEFORE the transaction completes is used as a
    feature. newbalanceOrig, newbalanceDest and isFlaggedFraud are NOT features.
    (--with-post-balances adds the two post-transaction balances for a
    sensitivity check only.)
  * Split first (70/15/15, stratified). Preprocessing, SMOTE and every rule
    limit are fitted on the training set only. The test set is used once.
  * Every number reported in Chapter Five must come from the files written to
    the output folder.
"""

import argparse, json, os, warnings
import numpy as np, pandas as pd
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve,
    roc_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from statsmodels.stats.contingency_tables import mcnemar

warnings.filterwarnings("ignore")

COLS = [
    "step",
    "type",
    "amount",
    "nameOrig",
    "oldbalanceOrg",
    "newbalanceOrig",
    "nameDest",
    "oldbalanceDest",
    "newbalanceDest",
    "isFraud",
    "isFlaggedFraud",
]


def parse():
    a = argparse.ArgumentParser()
    a.add_argument("--data", required=True, help="path to the PaySim CSV")
    a.add_argument("--out", default="results")
    a.add_argument("--seed", type=int, default=42)
    a.add_argument(
        "--boot",
        type=int,
        default=1000,
        help="bootstrap resamples (use 200 for a quick run)",
    )
    a.add_argument(
        "--tune-rows",
        type=int,
        default=300000,
        help="stratified training subsample used for the grid search of C (0 = use all)",
    )
    a.add_argument(
        "--smote-ratio",
        type=float,
        default=0.1,
        help="SMOTE target: fraud/legitimate ratio after resampling (training set only)",
    )
    a.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="optional stratified subsample of the whole file for a quick run (0 = all rows)",
    )
    a.add_argument(
        "--with-post-balances",
        action="store_true",
        help="sensitivity check: also use newbalanceOrig and newbalanceDest",
    )
    return a.parse_args()


# ----------------------------------------------------------------- data
def profile(raw, out):
    p = {
        "rows": int(len(raw)),
        "fraud": int(raw.isFraud.sum()),
        "fraud_pct": float(100 * raw.isFraud.mean()),
        "unique_nameOrig": int(raw.nameOrig.nunique()),
        "unique_nameDest": int(raw.nameDest.nunique()),
        "missing_values": int(raw.isna().sum().sum()),
        "isFlaggedFraud_rows": int(raw.isFlaggedFraud.sum()),
        "fraud_among_flagged": int(
            raw.loc[raw.isFlaggedFraud == 1, "isFraud"].sum()
        ),
        "steps": int(raw.step.max()),
    }
    by = (
        raw.groupby("type")
        .agg(rows=("isFraud", "size"), fraud=("isFraud", "sum"))
        .reset_index()
    )
    by["fraud_pct"] = 100 * by.fraud / by.rows
    by.to_csv(f"{out}/data_profile_by_type.csv", index=False)
    json.dump(p, open(f"{out}/data_profile.json", "w"), indent=2)
    return p


def prepare(raw, with_post):
    """Row-level features: each uses only that row, so nothing leaks across the split."""
    d = pd.DataFrame(index=raw.index)
    d["y"] = raw.isFraud.astype(int)
    d["amount"] = raw.amount.astype(float)
    d["type"] = raw.type.astype(str)
    d["hour"] = (raw.step % 24).astype(float)  # simulated hour of day
    d["night"] = (d["hour"] <= 4).astype(float)  # 00:00 to 04:59
    old_org = raw.oldbalanceOrg.astype(float).clip(lower=0)
    old_dest = raw.oldbalanceDest.astype(float).clip(lower=0)
    d["log_amount"] = np.log1p(d["amount"])
    d["log_old_org"] = np.log1p(old_org)
    d["log_old_dest"] = np.log1p(old_dest)
    d["balance_ratio"] = np.clip(
        d["amount"] / np.maximum(old_org, 1.0), 0, 1
    )
    d["sweep"] = ((old_org > 0) & (d["amount"] >= old_org)).astype(
        float
    )  # sender account emptied
    d["dest_merchant"] = (
        raw.nameDest.astype(str).str.startswith("M").astype(float)
    )
    d["dest_empty"] = (old_dest == 0).astype(
        float
    )  # recipient account empty before
    d["dest"] = raw.nameDest.astype(
        "category"
    )  # only for the blacklist rule
    d["flagged"] = raw.isFlaggedFraud.astype(
        int
    )  # only for rule R5, never a feature
    if with_post:
        d["log_new_org"] = np.log1p(
            raw.newbalanceOrig.astype(float).clip(lower=0)
        )
        d["log_new_dest"] = np.log1p(
            raw.newbalanceDest.astype(float).clip(lower=0)
        )
    return d


def rule_params(tr):
    legit = tr[tr.y == 0]
    return {
        "amount_limit": float(legit["amount"].quantile(0.99)),
        "amount_median": float(tr["amount"].median()),
        "bad_dest": set(tr.loc[tr.y == 1, "dest"].astype(str)),
    }


def rules(d, p):
    r = pd.DataFrame(index=d.index)
    r["R1"] = (d["amount"] > p["amount_limit"]).astype(int)  # amount limit
    r["R2"] = (
        (d["sweep"] == 1) & d["type"].isin(["TRANSFER", "CASH_OUT"])
    ).astype(
        int
    )  # balance sweep
    r["R3"] = (
        d["dest"].astype(str).isin(p["bad_dest"]).astype(int)
    )  # destination blacklist
    r["R4"] = (
        (d["night"] == 1) & (d["amount"] > p["amount_median"])
    ).astype(
        int
    )  # night and high amount
    r["R5"] = (d["flagged"] == 1).astype(int)  # control shipped with PaySim
    return r


# -------------------------------------------------------------- metrics
def metrics(y, pred, score=None):
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "fpr": fp / (fp + tn) if (fp + tn) else 0.0,
        "accuracy": accuracy_score(y, pred),
        "TP": int(tp),
        "FP": int(fp),
        "TN": int(tn),
        "FN": int(fn),
    }
    if score is not None:
        m["auc"] = roc_auc_score(y, score)
    return m


def best_threshold(y, s):
    p, r, t = precision_recall_curve(y, s)
    f = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    i = int(np.nanargmax(f))
    return float(t[i]), float(f[i])


def boot_ci(y, pred, score, rng, n):
    vals = {
        k: []
        for k in ["precision", "recall", "f1", "fpr", "accuracy", "auc"]
    }
    idx = np.arange(len(y))
    for _ in range(n):
        b = rng.choice(idx, len(idx), replace=True)
        if y[b].sum() == 0 or y[b].sum() == len(b):
            continue
        m = metrics(y[b], pred[b], score[b])
        for k in vals:
            vals[k].append(m[k])
    return {
        k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for k, v in vals.items()
    }


def mcnemar_fraud(y, pred_a, pred_b):
    """Exact McNemar on the fraud cases: did method A / method B detect each one?"""
    f = y == 1
    a, b = pred_a[f] == 1, pred_b[f] == 1
    tbl = [
        [int((a & b).sum()), int((a & ~b).sum())],
        [int((~a & b).sum()), int((~a & ~b).sum())],
    ]
    res = mcnemar(tbl, exact=True)
    return {
        "a_only": tbl[0][1],
        "b_only": tbl[1][0],
        "p": float(res.pvalue),
    }


# ----------------------------------------------------------------- main
def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    raw = pd.read_csv(args.data, usecols=COLS)
    prof = profile(raw, args.out)
    raw = raw.drop_duplicates().dropna(
        subset=["isFraud", "amount", "type", "step"]
    )
    d = prepare(raw, args.with_post_balances)
    del raw
    if args.max_rows and args.max_rows < len(d):
        d, _ = train_test_split(
            d,
            train_size=args.max_rows,
            stratify=d.y,
            random_state=args.seed,
        )

    tr, tmp = train_test_split(
        d, test_size=0.30, stratify=d.y, random_state=args.seed
    )
    va, te = train_test_split(
        tmp, test_size=0.50, stratify=tmp.y, random_state=args.seed
    )
    params = rule_params(tr)

    num = [
        "log_amount",
        "log_old_org",
        "log_old_dest",
        "balance_ratio",
        "hour",
    ] + (["log_new_org", "log_new_dest"] if args.with_post_balances else [])
    flags = ["night", "sweep", "dest_merchant", "dest_empty"]
    cat = ["type"]
    feats = num + flags + cat
    pre = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="median")),
                        ("sc", StandardScaler()),
                    ]
                ),
                num,
            ),
            (
                "flag",
                SimpleImputer(strategy="constant", fill_value=0),
                flags,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ],
        sparse_threshold=0,
    )
    Xtr, ytr, Xva, yva, Xte, yte = (
        tr[feats],
        tr.y.values,
        va[feats],
        va.y.values,
        te[feats],
        te.y.values,
    )

    rows = [
        [
            n,
            len(x),
            int(x.y.sum()),
            int((x.y == 0).sum()),
            round(100 * x.y.mean(), 4),
        ]
        for n, x in [
            ("Training (original)", tr),
            ("Validation", va),
            ("Test", te),
        ]
    ]

    conditions = {
        "A: original data": dict(smote=False, cw=None),
        "B: SMOTE (training only)": dict(smote=True, cw=None),
        "C: class weighting": dict(smote=False, cw="balanced"),
    }
    cv = StratifiedKFold(5, shuffle=True, random_state=args.seed)
    if args.tune_rows and args.tune_rows < len(tr):
        tidx, _ = train_test_split(
            np.arange(len(tr)),
            train_size=args.tune_rows,
            stratify=ytr,
            random_state=args.seed,
        )
        Xtune, ytune = Xtr.iloc[tidx], ytr[tidx]
    else:
        Xtune, ytune = Xtr, ytr

    fitted, val_f1, thr, res, n_sm = {}, {}, {}, {}, None
    for name, cfg in conditions.items():
        steps = [("pre", pre)]
        if cfg["smote"]:
            steps.append(
                (
                    "smote",
                    SMOTE(
                        sampling_strategy=args.smote_ratio,
                        random_state=args.seed,
                    ),
                )
            )
        steps.append(
            (
                "clf",
                LogisticRegression(
                    max_iter=2000, class_weight=cfg["cw"], solver="lbfgs"
                ),
            )
        )
        pipe = ImbPipeline(steps)
        gs = GridSearchCV(
            pipe,
            {"clf__C": [0.01, 0.1, 1, 10, 100]},
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            refit=False,
        )
        gs.fit(Xtune, ytune)
        C = gs.cv_results_["params"][
            int(np.argmin(gs.cv_results_["rank_test_score"]))
        ]["clf__C"]
        model = (
            clone(pipe).set_params(clf__C=C).fit(Xtr, ytr)
        )  # refit on the FULL training set
        tau, f1v = best_threshold(yva, model.predict_proba(Xva)[:, 1])
        s = model.predict_proba(Xte)[:, 1]
        pred = (s >= tau).astype(int)
        fitted[name], thr[name], val_f1[name] = model, tau, f1v
        res[name] = dict(score=s, pred=pred, m=metrics(yte, pred, s), C=C)
        if cfg["smote"]:
            Xt = model.named_steps["pre"].transform(Xtr)
            _, ys = SMOTE(
                sampling_strategy=args.smote_ratio, random_state=args.seed
            ).fit_resample(Xt, ytr)
            n_sm = (
                "Training (after SMOTE)",
                len(ys),
                int(ys.sum()),
                int((ys == 0).sum()),
                round(100 * ys.mean(), 4),
            )
    if n_sm:
        rows.insert(1, list(n_sm))
    pd.DataFrame(
        rows, columns=["Set", "Records", "Fraud", "Legitimate", "Fraud %"]
    ).to_csv(f"{args.out}/table_5_3_partitions.csv", index=False)

    pd.DataFrame(
        [
            [
                n,
                thr[n],
                res[n]["C"],
                val_f1[n],
                *[
                    res[n]["m"][k]
                    for k in ["precision", "recall", "f1", "auc", "fpr"]
                ],
            ]
            for n in conditions
        ],
        columns=[
            "Condition",
            "Threshold",
            "C",
            "Val_F1",
            "Precision",
            "Recall",
            "F1",
            "AUC",
            "FPR",
        ],
    ).to_csv(f"{args.out}/table_5_4_conditions.csv", index=False)

    final = max(
        val_f1, key=val_f1.get
    )  # chosen on VALIDATION F1, never on the test set
    F = res[final]
    ci = boot_ci(yte, F["pred"], F["score"], rng, args.boot)
    pd.DataFrame(
        [
            [k, F["m"][k], *ci[k]]
            for k in ["precision", "recall", "f1", "auc", "fpr", "accuracy"]
        ],
        columns=["Metric", "Value", "CI_low", "CI_high"],
    ).to_csv(f"{args.out}/table_5_5_final_model.csv", index=False)
    m = F["m"]
    pd.DataFrame(
        [
            ["Actual fraud", m["TP"], m["FN"]],
            ["Actual legitimate", m["FP"], m["TN"]],
        ],
        columns=["", "Predicted fraud", "Predicted legitimate"],
    ).to_csv(f"{args.out}/table_5_6_confusion.csv", index=False)

    # rule-based baseline on the same test set
    R = rules(te, params)
    rpred = R.any(axis=1).astype(int).values
    rm = metrics(yte, rpred)
    pd.DataFrame(
        [
            [
                "Rule-based baseline",
                rm["precision"],
                rm["recall"],
                rm["f1"],
                rm["fpr"],
            ],
            [
                "Logistic regression",
                m["precision"],
                m["recall"],
                m["f1"],
                m["fpr"],
            ],
            [
                "Difference (LR minus rules)",
                m["precision"] - rm["precision"],
                m["recall"] - rm["recall"],
                m["f1"] - rm["f1"],
                m["fpr"] - rm["fpr"],
            ],
        ],
        columns=["Method", "Precision", "Recall", "F1", "FPR"],
    ).to_csv(f"{args.out}/table_5_7_comparison.csv", index=False)
    pd.DataFrame(
        [
            [c, int(R[c].sum()), int(((R[c] == 1) & (yte == 1)).sum())]
            for c in R.columns
        ],
        columns=["Rule", "Times_fired", "Of_which_fraud"],
    ).to_csv(f"{args.out}/table_5_8_rules.csv", index=False)

    # odds ratios
    mdl = fitted[final]
    names = mdl.named_steps["pre"].get_feature_names_out()
    coef = mdl.named_steps["clf"].coef_[0]
    pd.DataFrame(
        {"Feature": names, "Coefficient": coef, "Odds_ratio": np.exp(coef)}
    ).sort_values("Coefficient", key=abs, ascending=False).to_csv(
        f"{args.out}/table_5_9_odds_ratios.csv", index=False
    )

    # hypothesis tests
    h1 = mcnemar_fraud(yte, F["pred"], rpred)
    h3 = mcnemar_fraud(
        yte,
        res["B: SMOTE (training only)"]["pred"],
        res["A: original data"]["pred"],
    )
    diffs, idx = [], np.arange(len(yte))
    for _ in range(args.boot):
        b = rng.choice(idx, len(idx), replace=True)
        if yte[b].sum() == 0:
            continue
        diffs.append(
            f1_score(yte[b], F["pred"][b], zero_division=0)
            - f1_score(yte[b], rpred[b], zero_division=0)
        )
    lo, hi = float(np.percentile(diffs, 2.5)), float(
        np.percentile(diffs, 97.5)
    )
    hyp = [
        [
            "H1",
            "LR recall > rule-based recall",
            f"McNemar exact p = {h1['p']:.4g} (LR-only detections {h1['a_only']}, rules-only {h1['b_only']})",
            (
                "Reject H0"
                if (h1["p"] < 0.05 and m["recall"] > rm["recall"])
                else "Fail to reject H0"
            ),
        ],
        [
            "H2",
            "LR F1 > rule-based F1",
            f"Paired bootstrap 95% CI of difference [{lo:.4f}, {hi:.4f}]",
            "Reject H0" if lo > 0 else "Fail to reject H0",
        ],
        [
            "H3",
            "SMOTE improves recall vs original data",
            f"McNemar exact p = {h3['p']:.4g} (SMOTE-only {h3['a_only']}, original-only {h3['b_only']})",
            (
                "Reject H0"
                if (
                    h3["p"] < 0.05
                    and res["B: SMOTE (training only)"]["m"]["recall"]
                    > res["A: original data"]["m"]["recall"]
                )
                else "Fail to reject H0"
            ),
        ],
    ]
    pd.DataFrame(
        hyp, columns=["ID", "Hypothesis", "Test_and_result", "Decision"]
    ).to_csv(f"{args.out}/table_5_10_hypotheses.csv", index=False)

    # figures
    fpr, tpr, _ = roc_curve(yte, F["score"])
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"Logistic regression (AUC = {m['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False-positive rate")
    plt.ylabel("True-positive rate")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_5_1_roc.png", dpi=200)
    plt.close()
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    plt.figure(figsize=(4.5, 4))
    share = cm / cm.sum(axis=1, keepdims=True)  # shade by share of each actual class
    plt.imshow(share, cmap="Blues", vmin=0, vmax=1)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(
            j, i, f"{v:,}\n({100 * share[i, j]:.1f}% of row)", ha="center",
            va="center", color="white" if share[i, j] > 0.5 else "black",
        )
    plt.xticks([0, 1], ["Legitimate", "Fraud"])
    plt.yticks([0, 1], ["Legitimate", "Fraud"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_5_2_confusion.png", dpi=200)
    plt.close()
    x = np.arange(4)
    w = 0.38
    plt.figure(figsize=(6, 4))
    plt.bar(
        x - w / 2,
        [rm[k] for k in ["precision", "recall", "f1", "fpr"]],
        w,
        label="Rule-based baseline",
    )
    plt.bar(
        x + w / 2,
        [m[k] for k in ["precision", "recall", "f1", "fpr"]],
        w,
        label="Logistic regression",
    )
    plt.xticks(x, ["Precision", "Recall", "F1", "FPR"])
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_5_3_comparison.png", dpi=200)
    plt.close()

    # ---- further analysis (Chapter Five, Section 5.7)
    n_te = len(yte)
    te_type = te["type"].values
    rows = []
    for t in sorted(set(te_type)):
        mk = te_type == t
        yt = yte[mk]
        row = [t, int(mk.sum()), int(yt.sum())]
        for p in (F["pred"][mk], rpred[mk]):
            tp = int(((p == 1) & (yt == 1)).sum())
            fp = int(((p == 1) & (yt == 0)).sum())
            row += [
                tp,
                fp,
                tp / yt.sum() if yt.sum() else float("nan"),
                tp / (tp + fp) if (tp + fp) else float("nan"),
            ]
        rows.append(row)
    pd.DataFrame(
        rows,
        columns=[
            "Type",
            "Records",
            "Fraud",
            "LR_TP",
            "LR_FP",
            "LR_recall",
            "LR_precision",
            "Rules_TP",
            "Rules_FP",
            "Rules_recall",
            "Rules_precision",
        ],
    ).to_csv(f"{args.out}/table_5_13_by_type.csv", index=False)

    rows = []
    for th in sorted(
        set([0.1, 0.25, 0.5, 0.75, 0.9, round(thr[final], 4)])
    ):
        p = (F["score"] >= th).astype(int)
        mm = metrics(yte, p)
        rows.append(
            [
                th,
                mm["precision"],
                mm["recall"],
                mm["f1"],
                mm["fpr"],
                mm["TP"] + mm["FP"],
                "chosen" if abs(th - round(thr[final], 4)) < 1e-9 else "",
            ]
        )
    pd.DataFrame(
        rows,
        columns=[
            "Threshold",
            "Precision",
            "Recall",
            "F1",
            "FPR",
            "Alerts",
            "Note",
        ],
    ).to_csv(
        f"{args.out}/table_5_14_threshold_sensitivity.csv", index=False
    )

    k = 100000 / n_te
    pd.DataFrame(
        [
            [
                "Rule-based baseline",
                (rm["TP"] + rm["FP"]) * k,
                rm["TP"] * k,
                rm["FP"] * k,
                rm["FN"] * k,
            ],
            [
                "Logistic regression",
                (m["TP"] + m["FP"]) * k,
                m["TP"] * k,
                m["FP"] * k,
                m["FN"] * k,
            ],
        ],
        columns=[
            "Method",
            "Alerts_per_100k",
            "Fraud_caught_per_100k",
            "False_alarms_per_100k",
            "Fraud_missed_per_100k",
        ],
    ).to_csv(f"{args.out}/table_5_15_workload.csv", index=False)

    pp, rr, _ = precision_recall_curve(yte, F["score"])
    plt.figure(figsize=(5, 4))
    plt.plot(rr, pp, label="Logistic regression")
    plt.scatter(
        [rm["recall"]],
        [rm["precision"]],
        color="C3",
        zorder=3,
        label="Rule-based baseline",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.ylim(0, 1.02)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_5_5_precision_recall.png", dpi=200)
    plt.close()

    top = (
        pd.DataFrame({"f": names, "c": coef})
        .reindex(pd.Series(coef).abs().sort_values(ascending=False).index)
        .head(10)
        .iloc[::-1]
    )
    plt.figure(figsize=(6, 4))
    plt.barh(top.f, top.c, color=["C3" if v > 0 else "C0" for v in top.c])
    plt.axvline(0, color="grey", lw=0.8)
    plt.xlabel("Coefficient (log-odds per standardised unit)")
    plt.tight_layout()
    plt.savefig(f"{args.out}/fig_5_4_coefficients.png", dpi=200)
    plt.close()

    # artefact for the prototype: model + preprocessing + threshold travel together
    joblib.dump(
        {
            "pipeline": mdl,
            "threshold": thr[final],
            "version": "lr-v1",
            "condition": final,
            "features": feats,
        },
        f"{args.out}/fraud_lr_v1.joblib",
    )
    summary = {
        "seed": args.seed,
        "final_condition": final,
        "threshold": thr[final],
        "records_used": int(len(d)),
        "post_balances_used": bool(args.with_post_balances),
        "smote_ratio": args.smote_ratio,
        "tune_rows": int(len(Xtune)),
        "data_profile": prof,
        "test_metrics": m,
        "baseline_metrics": rm,
        "rule_params": {
            "amount_limit": params["amount_limit"],
            "amount_median": params["amount_median"],
            "blacklisted_destinations": len(params["bad_dest"]),
        },
    }
    json.dump(
        summary,
        open(f"{args.out}/results.json", "w"),
        indent=2,
        default=float,
    )
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
