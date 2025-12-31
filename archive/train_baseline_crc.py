#!/usr/bin/env python3
import os
import argparse
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def pick_threshold_at_fpr(y_true, y_prob, target_fpr=0.05):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)

    order = np.argsort(-y_prob)
    y_true_sorted = y_true[order]
    y_prob_sorted = y_prob[order]

    neg_total = (y_true_sorted == 0).sum()
    if neg_total == 0:
        return None

    fp = 0
    tp = 0
    best_thr = None
    best_recall = None
    best_prec = None

    for i in range(len(y_true_sorted)):
        if y_true_sorted[i] == 0:
            fp += 1
        else:
            tp += 1

        fpr = fp / neg_total
        if fpr <= target_fpr:
            best_thr = y_prob_sorted[i]
            denom = (tp + fp)
            best_prec = (tp / denom) if denom else 0.0
            pos_total = (y_true_sorted == 1).sum()
            best_recall = (tp / pos_total) if pos_total else 0.0
        else:
            break

    return best_thr, best_prec, best_recall

def patient_split(df, seed=7, train=0.7, val=0.15, test=0.15):
    assert abs(train + val + test - 1.0) < 1e-6

    pats = df[["PATIENT", "COHORT"]].drop_duplicates()
    cases = pats[pats["COHORT"] == "case"]["PATIENT"].tolist()
    ctrls = pats[pats["COHORT"] == "control"]["PATIENT"].tolist()

    rng = np.random.default_rng(seed)
    rng.shuffle(cases)
    rng.shuffle(ctrls)

    def split_list(xs):
        n = len(xs)
        n_train = int(round(n * train))
        n_val = int(round(n * val))
        train_ids = xs[:n_train]
        val_ids = xs[n_train:n_train + n_val]
        test_ids = xs[n_train + n_val:]
        return set(train_ids), set(val_ids), set(test_ids)

    c_tr, c_va, c_te = split_list(cases)
    k_tr, k_va, k_te = split_list(ctrls)

    train_ids = c_tr | k_tr
    val_ids   = c_va | k_va
    test_ids  = c_te | k_te

    return train_ids, val_ids, test_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_table", default=os.path.join("output", "crc_ml_table.csv"))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--target_fpr", type=float, default=0.05)
    args = ap.parse_args()

    df = pd.read_csv(args.ml_table)

    needed = {"PATIENT", "COHORT", "label_in_window"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"ml_table missing columns: {missing}. Found: {list(df.columns)}")

    # features: keep numeric columns only, exclude obvious non-features
    drop_cols = {"label_in_window", "PATIENT", "COHORT", "outcome_has_crc", "ANCHOR_DATE", "MONTH_START"}
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].copy()
    y = df["label_in_window"].astype(int)

    train_ids, val_ids, test_ids = patient_split(df, seed=args.seed)

    tr = df["PATIENT"].isin(train_ids)
    va = df["PATIENT"].isin(val_ids)
    te = df["PATIENT"].isin(test_ids)

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    print("rows:", len(df))
    print("patients:", df["PATIENT"].nunique())
    print("positives:", int(y.sum()))
    print("\nSplit rows:")
    print("train:", int(tr.sum()), "val:", int(va.sum()), "test:", int(te.sum()))
    print("Split positives:")
    print("train:", int(y_tr.sum()), "val:", int(y_va.sum()), "test:", int(y_te.sum()))

    numeric_features = feature_cols

    pre = ColumnTransformer(
        transformers=[
            ("num", Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]), numeric_features)
        ],
        remainder="drop"
    )

    # class_weight helps because positives are very rare
    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs"
    )

    model = Pipeline(steps=[("pre", pre), ("clf", clf)])
    model.fit(X_tr, y_tr)

    def eval_split(name, Xs, ys):
        prob = model.predict_proba(Xs)[:, 1]
        auroc = roc_auc_score(ys, prob) if len(np.unique(ys)) > 1 else float("nan")
        auprc = average_precision_score(ys, prob) if len(np.unique(ys)) > 1 else float("nan")
        out = {"auroc": auroc, "auprc": auprc}
        return prob, out

    prob_va, met_va = eval_split("val", X_va, y_va)
    prob_te, met_te = eval_split("test", X_te, y_te)

    print("\nValidation metrics:", met_va)
    print("Test metrics:", met_te)

    thr_info = pick_threshold_at_fpr(y_va, prob_va, target_fpr=args.target_fpr)
    if thr_info[0] is None:
        print("\nCould not compute threshold@FPR (no negatives?)")
        return

    thr, prec, rec = thr_info
    print(f"\nChosen threshold on VAL to keep FPR <= {args.target_fpr:.3f}: {thr:.6f}")
    print(f"VAL precision={prec:.4f} recall={rec:.4f}")

    # apply that threshold to TEST
    yhat_te = (prob_te >= thr).astype(int)
    fp = int(((yhat_te == 1) & (y_te == 0)).sum())
    tp = int(((yhat_te == 1) & (y_te == 1)).sum())
    fn = int(((yhat_te == 0) & (y_te == 1)).sum())
    tn = int(((yhat_te == 0) & (y_te == 0)).sum())

    test_fpr = fp / (fp + tn) if (fp + tn) else 0.0
    test_tpr = tp / (tp + fn) if (tp + fn) else 0.0
    test_prec = tp / (tp + fp) if (tp + fp) else 0.0

    print("\nTEST @ chosen threshold:")
    print("tp, fp, fn, tn =", tp, fp, fn, tn)
    print(f"test_fpr={test_fpr:.4f}  recall={test_tpr:.4f}  precision={test_prec:.4f}")

if __name__ == "__main__":
    main()
