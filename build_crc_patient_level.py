#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_table", default="output/crc_ml_table_h6.csv")
    ap.add_argument("--out_csv", default="output/crc_patient_table_h6.csv")
    ap.add_argument("--label", default="label_future_crc")
    ap.add_argument("--start_month", type=int, default=1)
    ap.add_argument("--end_month", type=int, default=6)
    ap.add_argument("--drop_cols_regex", default="")
    args = ap.parse_args()

    df = pd.read_csv(args.ml_table)

    df = df[df["MONTH_INDEX_BACK"].between(args.start_month, args.end_month)].copy()

    keep_meta = ["PATIENT", "COHORT", "outcome_has_crc"]
    y = df.groupby("PATIENT")["outcome_has_crc"].max().rename("label_patient").reset_index()
    drop_cols = set(keep_meta + [args.label, "ANCHOR_DATE", "MONTH_START"])
    num_cols = [c for c in df.columns if c not in drop_cols]

    if args.drop_cols_regex.strip():
        rx = re.compile(args.drop_cols_regex)
        num_cols = [c for c in num_cols if not rx.search(c)]

    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg = df.groupby("PATIENT")[num_cols].agg(["max", "min", "mean", "sum"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    meta = df.groupby("PATIENT")[["COHORT", "outcome_has_crc"]].first().reset_index()
    out = meta.merge(y, on="PATIENT", how="left").merge(agg, on="PATIENT", how="left")

    out.to_csv(args.out_csv, index=False)
    print("wrote", args.out_csv, "rows", len(out))
    print("window", args.start_month, "to", args.end_month)
    print(out["label_patient"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
