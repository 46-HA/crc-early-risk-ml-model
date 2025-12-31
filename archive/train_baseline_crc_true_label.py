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
from sklearn.metrics import roc_auc_score, average_precision_score

def patient_split(df, seed=7, train=0.7, val=0.15, test=0.15):
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_table", default="output/crc_ml_table_h6.csv")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    df = pd.read_csv(args.ml_table)

    label = "label_future_crc"

    drop_cols = {
        "PATIENT", "COHORT", "ANCHOR_DATE", "MONTH_START",
        "outcome_has_crc", label
    }

    features = [c for c in df.columns if c not in drop_cols]

    X = df[features]
    y = df[label].astype(int)

    train_ids, val_ids, test_ids = patient_split(df, seed=args.seed)

    tr = df["PATIENT"].isin(train_ids)
    va = df["PATIENT"].isin(val_ids)
    te = df["PATIENT"].isin(test_ids)

    X_tr, y_tr = X[tr], y[tr]
    X_va, y_va = X[va], y[va]
    X_te, y_te = X[te], y[te]

    print("Patients:", df["PATIENT"].nunique())
    print("Total rows:", len(df))
    print("Positive rows:", int(y.sum()))
    print("\nSplit positives:")
    print("Train:", int(y_tr.sum()), "Val:", int(y_va.sum()), "Test:", int(y_te.sum()))

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), features)
    ])

    model = Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            solver="lbfgs"
        ))
    ])

    model.fit(X_tr, y_tr)

    def eval_split(Xs, ys, name):
        prob = model.predict_proba(Xs)[:, 1]
        auroc = roc_auc_score(ys, prob)
        auprc = average_precision_score(ys, prob)
        print(f"{name} AUROC={auroc:.4f}  AUPRC={auprc:.4f}")

    print("\n--- Performance ---")
    eval_split(X_va, y_va, "VAL")
    eval_split(X_te, y_te, "TEST")

if __name__ == "__main__":
    main()
