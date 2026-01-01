#!/usr/bin/env python3
import argparse
import pandas as pd
import re

def parse_csv_list(s: str):
    return [p.strip() for p in (s or "").split(",") if p.strip()]

def safe_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def build_trend_features(df: pd.DataFrame, lab_cols, early_range=(7, 12), late_range=(1, 3)):
    early = df[df["MONTH_INDEX_BACK"].between(early_range[0], early_range[1])]
    late  = df[df["MONTH_INDEX_BACK"].between(late_range[0],  late_range[1])]

    feats = []
    for lab in lab_cols:
        if lab not in df.columns:
            continue
        e = early.groupby("PATIENT")[lab].mean()
        l = late.groupby("PATIENT")[lab].mean()
        delta = (l - e).rename(f"{lab}_delta_late{late_range[0]}-{late_range[1]}_early{early_range[0]}-{early_range[1]}")
        feats.append(delta)

    if not feats:
        return pd.DataFrame({"PATIENT": df["PATIENT"].unique()})

    return pd.concat(feats, axis=1).reset_index()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ml_table", default="output/crc_ml_table.csv")
    ap.add_argument("--ml_tables", default="", help="comma-separated list")
    ap.add_argument("--out_csv", default="output/crc_patient_table.csv")
    ap.add_argument("--label", default="label_in_window")
    ap.add_argument("--start_month", type=int, default=1)
    ap.add_argument("--end_month", type=int, default=12)
    ap.add_argument("--drop_cols_regex", default="")

    ap.add_argument("--trend_early_start", type=int, default=7)
    ap.add_argument("--trend_early_end", type=int, default=12)
    ap.add_argument("--trend_late_start", type=int, default=1)
    ap.add_argument("--trend_late_end", type=int, default=3)
    args = ap.parse_args()

    ml_tables = parse_csv_list(args.ml_tables) if args.ml_tables.strip() else [args.ml_table]
    dfs = [pd.read_csv(p) for p in ml_tables]
    df = pd.concat(dfs, ignore_index=True)

    if "MONTH_INDEX_BACK" not in df.columns:
        raise KeyError(f"MONTH_INDEX_BACK missing. Found: {list(df.columns)}")

    df = df[df["MONTH_INDEX_BACK"].between(args.start_month, args.end_month)].copy()

    keep_meta = ["PATIENT", "COHORT", "outcome_has_crc"]
    y = (
        df.groupby("PATIENT")["outcome_has_crc"]
        .max()
        .rename("label_patient")
        .reset_index()
    )

    drop_cols = set(keep_meta + [args.label, "ANCHOR_DATE", "MONTH_START", "MONTH_END"])
    num_cols = [c for c in df.columns if c not in drop_cols]

    if args.drop_cols_regex.strip():
        rx = re.compile(args.drop_cols_regex)
        num_cols = [c for c in num_cols if not rx.search(c)]

    df = safe_numeric(df, num_cols)

    # Standard aggregations
    agg = df.groupby("PATIENT")[num_cols].agg(["max", "min", "mean", "sum"])
    agg.columns = [f"{a}_{b}" for a, b in agg.columns]
    agg = agg.reset_index()

    # ANY features for sparse binary-ish columns
    bin_candidates = [
        "anemia_signal","gi_signal","repeat_enc_signal","stool_hgb_any",
        "family_history_crc","had_colonoscopy","had_fobt_fit","screening_any",
        "ct_abd_pelvis_any","gi_imaging_any"
    ]
    bin_cols = [c for c in bin_candidates if c in df.columns]
    any_feats = None
    if bin_cols:
        any_feats = (
            df.groupby("PATIENT")[bin_cols]
            .max()
            .add_prefix("any_")
            .reset_index()
        )

    # Trend features (late - early) for key labs
    trend_labs = [
        "hgb_min_value","hct_min_value","mcv_min_value","ferritin_min_value","iron_min_value",
        "hgb_mean_value","hct_mean_value","mcv_mean_value","ferritin_mean_value","iron_mean_value",
    ]
    df = safe_numeric(df, trend_labs)

    trend_df = build_trend_features(
        df,
        lab_cols=trend_labs,
        early_range=(args.trend_early_start, args.trend_early_end),
        late_range=(args.trend_late_start, args.trend_late_end),
    )

    meta = df.groupby("PATIENT")[["COHORT", "outcome_has_crc"]].first().reset_index()

    out = meta.merge(y, on="PATIENT", how="left").merge(agg, on="PATIENT", how="left")
    if any_feats is not None:
        out = out.merge(any_feats, on="PATIENT", how="left")
    out = out.merge(trend_df, on="PATIENT", how="left")

    out.to_csv(args.out_csv, index=False)
    print("wrote", args.out_csv, "rows", len(out))
    print("window", args.start_month, "to", args.end_month)
    print(out["label_patient"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
