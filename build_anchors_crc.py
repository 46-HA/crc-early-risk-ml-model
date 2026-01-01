#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd

CRC_DESC_REGEX_DEFAULT = (
    r"(?:colorectal cancer|colon cancer|rectal cancer|"
    r"malignant neoplasm of colon|malignant neoplasm of rectum|"
    r"primary malignant neoplasm of colon|"
    r"overlapping malignant neoplasm of colon)"
)

FILES_NEEDED = ["patients.csv", "conditions.csv", "encounters.csv", "observations.csv"]

def parse_data_dirs(args):
    # Backwards compatible:
    # - If --data_dirs is provided: use it
    # - Else fall back to --data_dir
    if args.data_dirs and args.data_dirs.strip():
        dirs = [x.strip() for x in args.data_dirs.split(",") if x.strip()]
    else:
        dirs = [args.data_dir]
    # Ensure they exist
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        raise FileNotFoundError(f"Missing data dirs: {missing}")
    return dirs

def read_concat_csv(data_dirs, filename):
    dfs = []
    used = []
    for d in data_dirs:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
            used.append(p)
    if not dfs:
        raise FileNotFoundError(f"Missing {filename} in any of: {data_dirs}")
    out = pd.concat(dfs, ignore_index=True)
    print(f"loaded {filename}: {len(out):,} rows from {len(used)} files")
    return out

def pick_date_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing date col, tried {candidates}, found {list(df.columns)}")

def to_dt_naive(s):
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None)

def month_start(ts):
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_all",
                    help="Single dataset directory (legacy). Ignored if --data_dirs is set.")
    ap.add_argument("--data_dirs", default="",
                    help="Comma-separated list of dataset dirs, e.g. data_10k,data_30k")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--lookback_months", type=int, default=24)
    ap.add_argument("--controls_per_case", type=int, default=1)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--crc_regex", default=CRC_DESC_REGEX_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    data_dirs = parse_data_dirs(args)
    print("DATA_DIRS:", data_dirs)

    # Load and concat across dirs
    patients = read_concat_csv(data_dirs, "patients.csv")
    conditions = read_concat_csv(data_dirs, "conditions.csv")
    encounters = read_concat_csv(data_dirs, "encounters.csv")
    observations = read_concat_csv(data_dirs, "observations.csv")

    if "Id" not in patients.columns:
        raise KeyError(f"patients.csv missing Id, found {list(patients.columns)}")

    cond_date = pick_date_col(conditions, ["START", "DATE", "STOP"])
    enc_date  = pick_date_col(encounters, ["START", "DATE", "STOP"])
    obs_date  = pick_date_col(observations, ["DATE", "START", "STOP"])

    conditions[cond_date] = to_dt_naive(conditions[cond_date])
    encounters[enc_date]  = to_dt_naive(encounters[enc_date])
    observations[obs_date]= to_dt_naive(observations[obs_date])

    crc_mask = conditions["DESCRIPTION"].fillna("").str.contains(
        args.crc_regex, flags=re.IGNORECASE, regex=True
    )
    crc_rows = conditions.loc[crc_mask].dropna(subset=[cond_date]).copy()

    print("Total condition rows:", len(conditions))
    print("CRC condition rows:", len(crc_rows))
    print("Unique CRC patients:", crc_rows["PATIENT"].nunique())

    if crc_rows.empty:
        print("no crc rows found, try widening --crc_regex")
        return

    anchors_cases = (
        crc_rows.groupby("PATIENT")[cond_date]
        .min()
        .reset_index()
        .rename(columns={cond_date: "ANCHOR_DATE"})
        .sort_values("ANCHOR_DATE")
    )
    anchors_cases["COHORT"] = "case"
    anchors_cases["outcome_has_crc"] = 1

    cases_path = os.path.join(args.out_dir, "anchors_crc_cases.csv")
    anchors_cases.to_csv(cases_path, index=False)
    print("wrote", cases_path, f"({len(anchors_cases)} rows)")

    patients_df = patients[["Id"]].copy()
    patients_df.rename(columns={"Id": "PATIENT"}, inplace=True)

    crc_patients = set(anchors_cases["PATIENT"].unique())
    candidates = patients_df[~patients_df["PATIENT"].isin(crc_patients)].copy()

    print("precomputing patient timelines (fast)...")

    d_enc = encounters[["PATIENT", enc_date]].dropna()
    d_cond = conditions[["PATIENT", cond_date]].dropna()
    d_obs = observations[["PATIENT", obs_date]].dropna()

    d_enc.columns = ["PATIENT", "DATE"]
    d_cond.columns = ["PATIENT", "DATE"]
    d_obs.columns = ["PATIENT", "DATE"]

    all_dates = pd.concat([d_enc, d_cond, d_obs], ignore_index=True)

    timeline_df = (
        all_dates.groupby("PATIENT")["DATE"]
        .agg(["min", "max"])
        .reset_index()
        .rename(columns={"min": "MIN_DATE", "max": "MAX_DATE"})
    )

    timeline = timeline_df.set_index("PATIENT")[["MIN_DATE", "MAX_DATE"]]

    cand_ids = candidates["PATIENT"].sample(frac=1.0, random_state=args.seed).tolist()

    used = set()
    controls = []
    need = len(anchors_cases) * args.controls_per_case

    for pid in cand_ids:
        if len(controls) >= need:
            break
        if pid in used:
            continue
        if pid not in timeline.index:
            continue

        t0 = timeline.loc[pid, "MIN_DATE"]
        t1 = timeline.loc[pid, "MAX_DATE"]
        if pd.isna(t0) or pd.isna(t1):
            continue

        anchor = pd.to_datetime(t1) - pd.DateOffset(months=1)
        if pd.isna(anchor):
            continue

        min_required = month_start(anchor) - pd.DateOffset(months=args.lookback_months)
        if pd.to_datetime(t0) > min_required:
            continue

        used.add(pid)
        controls.append({
            "PATIENT": pid,
            "ANCHOR_DATE": anchor,
            "COHORT": "control",
            "outcome_has_crc": 0
        })

    anchors_ctrl = pd.DataFrame(controls)
    ctrl_path = os.path.join(args.out_dir, "anchors_crc_controls.csv")
    anchors_ctrl.to_csv(ctrl_path, index=False)
    print("wrote", ctrl_path, f"({len(anchors_ctrl)} rows)")

    if len(anchors_ctrl) < need:
        print(f"warning: requested {need} controls but only found {len(anchors_ctrl)} eligible controls")
        print("try lowering --lookback_months or using fewer --controls_per_case")

    anchors_all = pd.concat([anchors_cases, anchors_ctrl], ignore_index=True)
    all_path = os.path.join(args.out_dir, "anchors_crc_all.csv")
    anchors_all.to_csv(all_path, index=False)
    print("wrote", all_path, f"({len(anchors_all)} rows)")
    print("\ncohort counts:\n", anchors_all["COHORT"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
