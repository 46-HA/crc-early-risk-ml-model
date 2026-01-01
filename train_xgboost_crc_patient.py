#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score, average_precision_score

def parse_csv_list(s: str):
    parts = [p.strip() for p in (s or "").split(",") if p.strip()]
    return parts

def patient_split(df, seed=7, train=0.7, val=0.15):
    pats = df[["PATIENT", "COHORT"]].drop_duplicates()
    cases = pats[pats["COHORT"] == "case"]["PATIENT"].tolist()
    ctrls = pats[pats["COHORT"] == "control"]["PATIENT"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    rng.shuffle(ctrls)

    def split(xs):
        n = len(xs)
        a = int(n * train)
        b = int(n * (train + val))
        return set(xs[:a]), set(xs[a:b]), set(xs[b:])

    c_tr, c_va, c_te = split(cases)
    k_tr, k_va, k_te = split(ctrls)
    return c_tr | k_tr, c_va | k_va, c_te | k_te

def find_threshold_at_fpr(y_true, y_prob, target_fpr=0.05):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    neg_total = int((y_true == 0).sum())
    pos_total = int((y_true == 1).sum())
    if neg_total == 0 or pos_total == 0:
        return None

    fp = 0
    tp = 0
    best = None

    for i in range(len(y_true)):
        if y_true[i] == 0:
            fp += 1
        else:
            tp += 1

        fpr = fp / neg_total
        if fpr <= target_fpr:
            thr = y_prob[i]
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / pos_total if pos_total else 0.0
            best = (thr, prec, rec, fpr, tp, fp)
        else:
            break

    return best

def confusion_at_threshold(y_true, y_prob, thr):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_hat = (y_prob >= thr).astype(int)

    tp = int(((y_hat == 1) & (y_true == 1)).sum())
    fp = int(((y_hat == 1) & (y_true == 0)).sum())
    fn = int(((y_hat == 0) & (y_true == 1)).sum())
    tn = int(((y_hat == 0) & (y_true == 0)).sum())

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    return tp, fp, fn, tn, prec, rec, fpr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default="output/crc_patient_table_h6.csv", help="(legacy) single table")
    ap.add_argument("--tables", default="", help="comma-separated list of patient tables")
    ap.add_argument("--label", default="label_patient")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    ap.add_argument("--n_estimators", type=int, default=1200)
    ap.add_argument("--max_depth", type=int, default=4)
    ap.add_argument("--learning_rate", type=float, default=0.03)
    ap.add_argument("--subsample", type=float, default=0.9)
    ap.add_argument("--colsample_bytree", type=float, default=0.9)
    ap.add_argument("--test_thresholds", default="")
    args = ap.parse_args()

    tables = parse_csv_list(args.tables) if args.tables.strip() else [args.table]

    print("loading tables:", tables, flush=True)
    dfs = [pd.read_csv(t) for t in tables]
    df = pd.concat(dfs, ignore_index=True)

    # de-dupe patients if the same PATIENT appears twice (prevents leakage)
    df = df.drop_duplicates(subset=["PATIENT"], keep="first").copy()

    drop = {"PATIENT", "COHORT", "outcome_has_crc", args.label}
    feat_cols = [c for c in df.columns if c not in drop]

    X = df[feat_cols]
    y = df[args.label].astype(int).values

    tr_ids, va_ids, te_ids = patient_split(df, seed=args.seed)
    tr = df["PATIENT"].isin(tr_ids).values
    va = df["PATIENT"].isin(va_ids).values
    te = df["PATIENT"].isin(te_ids).values

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    imp = SimpleImputer(strategy="median")
    X_tr_i = imp.fit_transform(X_tr)
    X_va_i = imp.transform(X_va)
    X_te_i = imp.transform(X_te)

    from xgboost import XGBClassifier

    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    spw = (neg / pos) if pos else 1.0

    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_lambda=1.0,
        min_child_weight=1.0,
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=args.seed,
        scale_pos_weight=spw,
    )

    clf.fit(X_tr_i, y_tr)

    p_va = clf.predict_proba(X_va_i)[:, 1]
    p_te = clf.predict_proba(X_te_i)[:, 1]

    print("VAL AUROC=%.4f  AUPRC=%.4f" % (roc_auc_score(y_va, p_va), average_precision_score(y_va, p_va)), flush=True)
    print("TEST AUROC=%.4f  AUPRC=%.4f" % (roc_auc_score(y_te, p_te), average_precision_score(y_te, p_te)), flush=True)

    best = find_threshold_at_fpr(y_va, p_va, target_fpr=args.target_fpr)
    if best is None:
        print("cannot compute threshold", flush=True)
        return

    thr, val_prec, val_rec, val_fpr, val_tp, val_fp = best
    print("VAL threshold@fpr<=%.3f thr=%.6f prec=%.4f rec=%.4f fpr=%.4f tp=%d fp=%d" %
          (args.target_fpr, thr, val_prec, val_rec, val_fpr, val_tp, val_fp), flush=True)

    tp, fp, fn, tn, prec, rec, fpr = confusion_at_threshold(y_te, p_te, thr)
    print("TEST @thr prec=%.4f rec=%.4f fpr=%.4f tp=%d fp=%d fn=%d tn=%d" %
          (prec, rec, fpr, tp, fp, fn, tn), flush=True)

    extra = [thr, 0.92, 0.94, 0.96, 0.98]
    if args.test_thresholds.strip():
        try:
            extra = [float(x) for x in args.test_thresholds.split(",") if x.strip()]
        except:
            extra = [thr, 0.92, 0.94, 0.96, 0.98]

    print("TEST_THRESHOLDS", flush=True)
    for t in extra:
        tp, fp, fn, tn, prec, rec, fpr = confusion_at_threshold(y_te, p_te, t)
        print("TEST thr=%.4f prec=%.4f rec=%.4f fpr=%.4f tp=%d fp=%d fn=%d tn=%d" %
              (t, prec, rec, fpr, tp, fp, fn, tn), flush=True)

if __name__ == "__main__":
    main()
