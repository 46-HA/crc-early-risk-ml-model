#!/usr/bin/env python3
import os
import re
import argparse
import pandas as pd

GI_DESC_REGEX_DEFAULT = (
    r"(?:abdominal pain|blood in stool|hematochezia|melena|diarrhea|"
    r"constipation|weight loss|nausea|vomiting|change in bowel|"
    r"fecal occult blood|occult blood|rectal bleeding|"
    r"iron deficiency anemia|anemia)"
)
HGB_DESC_REGEX_DEFAULT = r"(?:hemoglobin|hgb)"

def pick_date_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing date col, tried {candidates}, found {list(df.columns)}")

def to_dt_naive(s):
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--anchors_csv", default=os.path.join("output", "anchors_crc_all.csv"))
    ap.add_argument("--lookback_months", type=int, default=24)
    ap.add_argument("--hgb_low_threshold", type=float, default=11.0)
    ap.add_argument("--gi_regex", default=GI_DESC_REGEX_DEFAULT)
    ap.add_argument("--hgb_regex", default=HGB_DESC_REGEX_DEFAULT)
    ap.add_argument("--label_horizon_months", type=int, default=6)
    ap.add_argument("--include_anchor_month", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.anchors_csv):
        raise FileNotFoundError(f"Missing anchors file: {args.anchors_csv}. Run build_anchors_crc.py first.")

    anchors = pd.read_csv(args.anchors_csv)
    need_cols = {"PATIENT", "ANCHOR_DATE", "COHORT", "outcome_has_crc"}
    missing = need_cols - set(anchors.columns)
    if missing:
        raise KeyError(f"anchors missing {missing}. Found: {list(anchors.columns)}")

    anchors["ANCHOR_DATE"] = pd.to_datetime(anchors["ANCHOR_DATE"], errors="coerce")
    anchors = anchors.dropna(subset=["ANCHOR_DATE"]).copy()
    anchors["ANCHOR_YM"] = anchors["ANCHOR_DATE"].dt.year * 12 + anchors["ANCHOR_DATE"].dt.month

    conditions = pd.read_csv(os.path.join(args.data_dir, "conditions.csv"))
    encounters = pd.read_csv(os.path.join(args.data_dir, "encounters.csv"))
    observations = pd.read_csv(os.path.join(args.data_dir, "observations.csv"))

    cond_date = pick_date_col(conditions, ["START", "DATE", "STOP"])
    enc_date  = pick_date_col(encounters, ["START", "DATE", "STOP"])
    obs_date  = pick_date_col(observations, ["DATE", "START", "STOP"])

    conditions[cond_date]  = to_dt_naive(conditions[cond_date])
    encounters[enc_date]   = to_dt_naive(encounters[enc_date])
    observations[obs_date] = to_dt_naive(observations[obs_date])

    enc = encounters[["PATIENT", enc_date]].dropna().copy()
    enc["YM"] = (enc[enc_date].dt.year * 12 + enc[enc_date].dt.month).astype("int64")
    enc_m = enc.groupby(["PATIENT", "YM"]).size().rename("encounter_count").reset_index()

    gi_mask = conditions["DESCRIPTION"].fillna("").str.contains(args.gi_regex, flags=re.IGNORECASE, regex=True)
    gi = conditions.loc[gi_mask, ["PATIENT", cond_date]].dropna().copy()
    gi["YM"] = (gi[cond_date].dt.year * 12 + gi[cond_date].dt.month).astype("int64")
    gi_m = gi.groupby(["PATIENT", "YM"]).size().rename("gi_condition_count").reset_index()

    hgb_mask = observations["DESCRIPTION"].fillna("").str.contains(args.hgb_regex, flags=re.IGNORECASE, regex=True)
    hgb = observations.loc[hgb_mask, ["PATIENT", obs_date, "VALUE"]].dropna(subset=[obs_date]).copy()
    hgb["VALUE_NUM"] = pd.to_numeric(hgb.get("VALUE"), errors="coerce")
    hgb["YM"] = (hgb[obs_date].dt.year * 12 + hgb[obs_date].dt.month).astype("int64")

    hgb_m = (
        hgb.groupby(["PATIENT", "YM"])
        .agg(
            hgb_obs_count=("VALUE_NUM", "count"),
            hgb_min_value=("VALUE_NUM", "min")
        )
        .reset_index()
    )

    feats = enc_m.merge(gi_m, on=["PATIENT", "YM"], how="outer").merge(hgb_m, on=["PATIENT", "YM"], how="outer")
    feats["encounter_count"] = feats["encounter_count"].fillna(0).astype(int)
    feats["gi_condition_count"] = feats["gi_condition_count"].fillna(0).astype(int)
    feats["hgb_obs_count"] = feats["hgb_obs_count"].fillna(0).astype(int)

    feats["gi_signal"] = (feats["gi_condition_count"] > 0).astype(int)
    feats["repeat_enc_signal"] = (feats["encounter_count"] >= 2).astype(int)
    feats["anemia_signal"] = ((feats["hgb_min_value"].notna()) & (feats["hgb_min_value"] < args.hgb_low_threshold)).astype(int)
    feats["signals_sum"] = feats["gi_signal"] + feats["repeat_enc_signal"] + feats["anemia_signal"]
    feats["escalation_positive"] = (feats["signals_sum"] >= 2).astype(int)

    feats_idx = feats.set_index(["PATIENT", "YM"])

    out_rows = []
    lo = 0 if args.include_anchor_month else 1

    for _, a in anchors.iterrows():
        pid = a["PATIENT"]
        anchor_ym = int(a["ANCHOR_YM"])
        cohort = a["COHORT"]
        outcome = int(a["outcome_has_crc"]) if pd.notna(a["outcome_has_crc"]) else 0
        is_case = outcome == 1

        for i in range(0, args.lookback_months + 1):
            ym = anchor_ym - i
            year = ym // 12
            month = ym % 12
            if month == 0:
                year -= 1
                month = 12
            m_start = pd.Timestamp(year=year, month=month, day=1)

            if (pid, ym) in feats_idx.index:
                r = feats_idx.loc[(pid, ym)]
                enc_count = int(r["encounter_count"])
                gi_count = int(r["gi_condition_count"])
                hgb_count = int(r["hgb_obs_count"])
                hgb_min = r["hgb_min_value"]
                gi_signal = int(r["gi_signal"])
                rep_signal = int(r["repeat_enc_signal"])
                anemia_signal = int(r["anemia_signal"])
                signals_sum = int(r["signals_sum"])
                escalation_pos = int(r["escalation_positive"])
            else:
                enc_count = gi_count = hgb_count = 0
                hgb_min = None
                gi_signal = rep_signal = anemia_signal = 0
                signals_sum = 0
                escalation_pos = 0

            months_to_crc = i
            if not is_case:
                label_future = 0
            else:
                label_future = int(lo <= months_to_crc <= args.label_horizon_months)

            out_rows.append({
                "PATIENT": pid,
                "COHORT": cohort,
                "outcome_has_crc": outcome,
                "ANCHOR_DATE": a["ANCHOR_DATE"].date() if hasattr(a["ANCHOR_DATE"], "date") else a["ANCHOR_DATE"],
                "MONTH_INDEX_BACK": i,
                "MONTH_START": m_start.date(),
                "months_to_crc": int(months_to_crc),
                "anemia_signal": anemia_signal,
                "gi_signal": gi_signal,
                "repeat_enc_signal": rep_signal,
                "signals_sum": signals_sum,
                "escalation_positive": escalation_pos,
                "gi_condition_count": gi_count,
                "encounter_count": enc_count,
                "hgb_obs_count": hgb_count,
                "hgb_min_value": float(hgb_min) if pd.notna(hgb_min) else None,
                "label_future_crc": label_future,
            })

    df = pd.DataFrame(out_rows).sort_values(["PATIENT", "MONTH_INDEX_BACK"])

    df["gi_persist_3m"] = (
        df.groupby("PATIENT")["gi_signal"]
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["encounter_delta"] = (
        df.groupby("PATIENT")["encounter_count"]
        .diff()
        .fillna(0)
    )

    df["hgb_delta"] = (
        df.groupby("PATIENT")["hgb_min_value"]
        .diff()
        .fillna(0)
    )

    df["encounter_increasing"] = (df["encounter_delta"] > 0).astype(int)
    df["hgb_dropping"] = (df["hgb_delta"] < 0).astype(int)

    out_name = f"crc_ml_table_h{args.label_horizon_months}.csv"
    out_path = os.path.join(args.out_dir, out_name)
    df.to_csv(out_path, index=False)

    print("wrote", out_path, f"({len(df)} rows)")
    print(df["label_future_crc"].value_counts(dropna=False))
    print(df.groupby("COHORT")["label_future_crc"].sum())

if __name__ == "__main__":
    main()
