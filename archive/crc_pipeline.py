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

def month_start(ts):
    return pd.Timestamp(year=ts.year, month=ts.month, day=1)

def iter_month_bins(anchor, lookback_months):
    a0 = month_start(anchor)
    for i in range(0, lookback_months + 1):
        m_start = a0 - pd.DateOffset(months=i)
        m_end = (a0 - pd.DateOffset(months=i-1)) if i > 0 else (a0 + pd.DateOffset(months=1))
        yield i, m_start, m_end

def first_two_consecutive_true(flags):
    for i in range(len(flags) - 1):
        if flags[i] and flags[i+1]:
            return i
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--anchors_csv", default=os.path.join("output", "anchors_crc_all.csv"))
    ap.add_argument("--lookback_months", type=int, default=24)
    ap.add_argument("--hgb_low_threshold", type=float, default=11.0)
    ap.add_argument("--gi_regex", default=GI_DESC_REGEX_DEFAULT)
    ap.add_argument("--hgb_regex", default=HGB_DESC_REGEX_DEFAULT)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.anchors_csv):
        raise FileNotFoundError(
            f"Missing anchors file: {args.anchors_csv}\n"
            f"Run: python build_anchors_crc.py --lookback_months {args.lookback_months}"
        )

    anchors_all = pd.read_csv(args.anchors_csv)
    if "PATIENT" not in anchors_all.columns or "ANCHOR_DATE" not in anchors_all.columns:
        raise KeyError(f"anchors_csv must contain PATIENT, ANCHOR_DATE. Found: {list(anchors_all.columns)}")

    anchors_all["ANCHOR_DATE"] = pd.to_datetime(anchors_all["ANCHOR_DATE"], errors="coerce")
    anchors_all = anchors_all.dropna(subset=["ANCHOR_DATE"]).copy()

    conditions = pd.read_csv(os.path.join(args.data_dir, "conditions.csv"))
    encounters = pd.read_csv(os.path.join(args.data_dir, "encounters.csv"))
    observations = pd.read_csv(os.path.join(args.data_dir, "observations.csv"))

    cond_date = pick_date_col(conditions, ["START", "DATE", "STOP"])
    enc_date  = pick_date_col(encounters, ["START", "DATE", "STOP"])
    obs_date  = pick_date_col(observations, ["DATE", "START", "STOP"])

    conditions[cond_date] = to_dt_naive(conditions[cond_date])
    encounters[enc_date]  = to_dt_naive(encounters[enc_date])
    observations[obs_date]= to_dt_naive(observations[obs_date])

    gi_mask = conditions["DESCRIPTION"].fillna("").str.contains(args.gi_regex, flags=re.IGNORECASE, regex=True)
    gi_conds = conditions.loc[gi_mask].dropna(subset=[cond_date]).copy()

    hgb_mask = observations["DESCRIPTION"].fillna("").str.contains(args.hgb_regex, flags=re.IGNORECASE, regex=True)
    hgb_obs = observations.loc[hgb_mask].dropna(subset=[obs_date]).copy()
    hgb_obs["VALUE_NUM"] = pd.to_numeric(hgb_obs.get("VALUE"), errors="coerce")

    monthly_rows = []
    window_rows = []

    have_cohort = "COHORT" in anchors_all.columns
    have_outcome = "outcome_has_crc" in anchors_all.columns

    for _, ar in anchors_all.iterrows():
        pid = ar["PATIENT"]
        anchor = ar["ANCHOR_DATE"]
        if pd.isna(anchor):
            continue

        cohort = ar["COHORT"] if have_cohort else None
        outcome = int(ar["outcome_has_crc"]) if have_outcome and pd.notna(ar["outcome_has_crc"]) else 0
        is_case = outcome == 1

        feats_newest_first = []

        for i, m_start, m_end in iter_month_bins(anchor, args.lookback_months):
            gi_pid = gi_conds[(gi_conds["PATIENT"] == pid) & (gi_conds[cond_date] >= m_start) & (gi_conds[cond_date] < m_end)]
            enc_pid = encounters[(encounters["PATIENT"] == pid) & (encounters[enc_date] >= m_start) & (encounters[enc_date] < m_end)]
            hgb_pid = hgb_obs[(hgb_obs["PATIENT"] == pid) & (hgb_obs[obs_date] >= m_start) & (hgb_obs[obs_date] < m_end)]

            gi_signal = int(len(gi_pid) > 0)
            anemia_signal = int((hgb_pid["VALUE_NUM"].dropna() < args.hgb_low_threshold).any())
            repeat_enc_signal = int(len(enc_pid) >= 2)

            signals_sum = gi_signal + anemia_signal + repeat_enc_signal
            escalation_positive = int(signals_sum >= 2)

            hgb_min_value = None
            if hgb_pid["VALUE_NUM"].notna().any():
                hgb_min_value = float(hgb_pid["VALUE_NUM"].min())

            row = {
                "PATIENT": pid,
                "ANCHOR_DATE": anchor.date() if hasattr(anchor, "date") else anchor,
                "MONTH_INDEX_BACK": i,
                "MONTH_START": m_start.date(),
                "MONTH_END": m_end.date(),
                "anemia_signal": anemia_signal,
                "gi_signal": gi_signal,
                "repeat_enc_signal": repeat_enc_signal,
                "signals_sum": signals_sum,
                "escalation_positive": escalation_positive,
                "gi_condition_count": int(len(gi_pid)),
                "encounter_count": int(len(enc_pid)),
                "hgb_obs_count": int(len(hgb_pid)),
                "hgb_min_value": hgb_min_value,
            }
            if have_cohort:
                row["COHORT"] = cohort
            if have_outcome:
                row["outcome_has_crc"] = outcome

            feats_newest_first.append(row)

        monthly_rows.extend(feats_newest_first)

        if not is_case:
            window_rows.append({
                "PATIENT": pid,
                "ANCHOR_DATE": anchor.date() if hasattr(anchor, "date") else anchor,
                "window_found": 0,
                "window_month1_start": None,
                "window_month2_start": None,
                "window_month1_index_back": None,
                "window_month2_index_back": None,
            })
        else:
            feats_oldest_first = list(reversed(feats_newest_first))
            flags = [bool(x["escalation_positive"]) for x in feats_oldest_first]
            idx = first_two_consecutive_true(flags)

            if idx is None:
                window_rows.append({
                    "PATIENT": pid,
                    "ANCHOR_DATE": anchor.date() if hasattr(anchor, "date") else anchor,
                    "window_found": 0,
                    "window_month1_start": None,
                    "window_month2_start": None,
                    "window_month1_index_back": None,
                    "window_month2_index_back": None,
                })
            else:
                m1 = feats_oldest_first[idx]
                m2 = feats_oldest_first[idx+1]
                window_rows.append({
                    "PATIENT": pid,
                    "ANCHOR_DATE": anchor.date() if hasattr(anchor, "date") else anchor,
                    "window_found": 1,
                    "window_month1_start": m1["MONTH_START"],
                    "window_month2_start": m2["MONTH_START"],
                    "window_month1_index_back": m1["MONTH_INDEX_BACK"],
                    "window_month2_index_back": m2["MONTH_INDEX_BACK"],
                })

    monthly_df = pd.DataFrame(monthly_rows)
    windows_df = pd.DataFrame(window_rows)

    monthly_df = monthly_df.sort_values(["PATIENT", "MONTH_INDEX_BACK"])
    monthly_path = os.path.join(args.out_dir, "crc_monthly_features.csv")
    monthly_df.to_csv(monthly_path, index=False)

    windows_path = os.path.join(args.out_dir, "crc_windows.csv")
    windows_df.to_csv(windows_path, index=False)

    print("wrote", monthly_path, f"({len(monthly_df)} rows)")
    print("wrote", windows_path, f"({len(windows_df)} rows)")
    print("window_found counts:\n", windows_df["window_found"].value_counts(dropna=False))

    feats = pd.read_csv(monthly_path)
    wins = pd.read_csv(windows_path)

    feats["MONTH_START"] = pd.to_datetime(feats["MONTH_START"], errors="coerce")
    feats["ANCHOR_DATE"] = pd.to_datetime(feats["ANCHOR_DATE"], errors="coerce")

    wins["window_month1_start"] = pd.to_datetime(wins["window_month1_start"], errors="coerce")
    wins["window_month2_start"] = pd.to_datetime(wins["window_month2_start"], errors="coerce")

    df = feats.merge(
        wins[["PATIENT", "window_found", "window_month1_start", "window_month2_start"]],
        on="PATIENT",
        how="left"
    )

    df["label_in_window"] = 0
    in_win = (df["window_found"] == 1) & (
        (df["MONTH_START"] == df["window_month1_start"]) |
        (df["MONTH_START"] == df["window_month2_start"])
    )
    df.loc[in_win, "label_in_window"] = 1

    if "outcome_has_crc" in df.columns:
        df.loc[df["outcome_has_crc"] == 0, "label_in_window"] = 0

    keep_cols = [
        "PATIENT",
        "COHORT" if "COHORT" in df.columns else None,
        "outcome_has_crc" if "outcome_has_crc" in df.columns else None,
        "ANCHOR_DATE", "MONTH_START", "MONTH_INDEX_BACK",
        "anemia_signal", "gi_signal", "repeat_enc_signal",
        "gi_condition_count", "encounter_count", "hgb_obs_count", "hgb_min_value",
        "label_in_window"
    ]
    keep_cols = [c for c in keep_cols if c is not None]

    df = df[keep_cols].sort_values(["PATIENT", "MONTH_INDEX_BACK"])

    ml_path = os.path.join(args.out_dir, "crc_ml_table.csv")
    df.to_csv(ml_path, index=False)
    print("wrote", ml_path, f"({len(df)} rows)")
    print("label_in_window counts:\n", df["label_in_window"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
