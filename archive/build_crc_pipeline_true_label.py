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

def pick_date_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError

def to_dt_naive(s):
    s = pd.to_datetime(s, errors="coerce", utc=True)
    return s.dt.tz_convert(None)

def add_obs_monthly(obs, obs_date, regex, name):
    m = obs["DESCRIPTION"].fillna("").str.contains(regex, flags=re.IGNORECASE, regex=True)
    t = obs.loc[m, ["PATIENT", obs_date, "VALUE"]].dropna(subset=[obs_date]).copy()
    t["VALUE_NUM"] = pd.to_numeric(t["VALUE"], errors="coerce")
    t = t.dropna(subset=["VALUE_NUM"])
    t["YM"] = t[obs_date].dt.year * 12 + t[obs_date].dt.month
    out = (
        t.groupby(["PATIENT", "YM"])
        .agg(**{
            f"{name}_count": ("VALUE_NUM", "count"),
            f"{name}_min": ("VALUE_NUM", "min"),
        })
        .reset_index()
    )
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--anchors_csv", default="output/anchors_crc_all.csv")
    ap.add_argument("--lookback_months", type=int, default=24)
    ap.add_argument("--hgb_low_threshold", type=float, default=11.0)
    ap.add_argument("--gi_regex", default=GI_DESC_REGEX_DEFAULT)
    ap.add_argument("--label_horizon_months", type=int, default=6)
    ap.add_argument("--include_anchor_month", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    patients = pd.read_csv(os.path.join(args.data_dir, "patients.csv"))
    patients["BIRTHDATE"] = pd.to_datetime(patients["BIRTHDATE"], errors="coerce")
    patients["is_male"] = (patients["GENDER"].astype(str).str.upper().str.startswith("M")).astype(int)
    patients = patients[["Id", "BIRTHDATE", "is_male"]].rename(columns={"Id": "PATIENT"}).set_index("PATIENT")

    anchors = pd.read_csv(args.anchors_csv)
    anchors["ANCHOR_DATE"] = pd.to_datetime(anchors["ANCHOR_DATE"], errors="coerce")
    anchors = anchors.dropna(subset=["ANCHOR_DATE"])
    anchors["ANCHOR_YM"] = anchors["ANCHOR_DATE"].dt.year * 12 + anchors["ANCHOR_DATE"].dt.month

    conditions = pd.read_csv(os.path.join(args.data_dir, "conditions.csv"))
    encounters = pd.read_csv(os.path.join(args.data_dir, "encounters.csv"))
    observations = pd.read_csv(os.path.join(args.data_dir, "observations.csv"))
    procedures = pd.read_csv(os.path.join(args.data_dir, "procedures.csv"))

    cond_date = pick_date_col(conditions, ["START", "DATE", "STOP"])
    enc_date = pick_date_col(encounters, ["START", "DATE", "STOP"])
    obs_date = pick_date_col(observations, ["DATE", "START", "STOP"])
    proc_date = pick_date_col(procedures, ["DATE", "START", "STOP"])

    conditions[cond_date] = to_dt_naive(conditions[cond_date])
    encounters[enc_date] = to_dt_naive(encounters[enc_date])
    observations[obs_date] = to_dt_naive(observations[obs_date])
    procedures[proc_date] = to_dt_naive(procedures[proc_date])

    enc = encounters[["PATIENT", enc_date]].dropna().copy()
    enc["YM"] = enc[enc_date].dt.year * 12 + enc[enc_date].dt.month
    enc_m = enc.groupby(["PATIENT", "YM"]).size().rename("encounter_count").reset_index()

    gi = conditions[
        conditions["DESCRIPTION"].fillna("").str.contains(args.gi_regex, flags=re.IGNORECASE, regex=True)
    ][["PATIENT", cond_date]].dropna().copy()
    gi["YM"] = gi[cond_date].dt.year * 12 + gi[cond_date].dt.month
    gi_m = gi.groupby(["PATIENT", "YM"]).size().rename("gi_condition_count").reset_index()

    col_mask = procedures["DESCRIPTION"].fillna("").str.contains(r"(?:colonoscopy)", flags=re.IGNORECASE, regex=True)
    col = procedures.loc[col_mask, ["PATIENT", proc_date]].dropna().copy()
    col["YM"] = col[proc_date].dt.year * 12 + col[proc_date].dt.month
    col_m = col.groupby(["PATIENT", "YM"]).size().rename("colonoscopy_count").reset_index()

    ob_mask = procedures["DESCRIPTION"].fillna("").str.contains(r"(?:occult blood in feces)", flags=re.IGNORECASE, regex=True)
    ob = procedures.loc[ob_mask, ["PATIENT", proc_date]].dropna().copy()
    ob["YM"] = ob[proc_date].dt.year * 12 + ob[proc_date].dt.month
    ob_m = ob.groupby(["PATIENT", "YM"]).size().rename("occult_blood_count").reset_index()

    hgb_blood = add_obs_monthly(
        observations, obs_date,
        r"(?:Hemoglobin \[Mass/volume\] in Blood)$",
        "hgb"
    )
    hct = add_obs_monthly(
        observations, obs_date,
        r"(?:Hematocrit \[Volume Fraction\] of Blood)",
        "hct"
    )
    mcv = add_obs_monthly(
        observations, obs_date,
        r"(?:MCV \[Entitic mean volume\] in Red Blood Cells)",
        "mcv"
    )
    ferr = add_obs_monthly(
        observations, obs_date,
        r"(?:Ferritin \[Mass/volume\] in Serum or Plasma)$",
        "ferritin"
    )
    iron = add_obs_monthly(
        observations, obs_date,
        r"(?:Iron \[Mass/volume\] in Serum or Plasma)$",
        "iron"
    )
    tibc = add_obs_monthly(
        observations, obs_date,
        r"(?:Iron binding capacity \[Mass/volume\] in Serum or Plasma)$",
        "tibc"
    )
    sat = add_obs_monthly(
        observations, obs_date,
        r"(?:Iron saturation \[Mass Fraction\] in Serum or Plasma)$",
        "iron_sat"
    )
    stool_hgb = add_obs_monthly(
        observations, obs_date,
        r"(?:Hemoglobin\.gastrointestinal\.lower \[Presence\] in Stool)",
        "stool_hgb"
    )

    feats = (
        enc_m.merge(gi_m, on=["PATIENT", "YM"], how="outer")
        .merge(col_m, on=["PATIENT", "YM"], how="outer")
        .merge(ob_m, on=["PATIENT", "YM"], how="outer")
        .merge(hgb_blood, on=["PATIENT", "YM"], how="outer")
        .merge(hct, on=["PATIENT", "YM"], how="outer")
        .merge(mcv, on=["PATIENT", "YM"], how="outer")
        .merge(ferr, on=["PATIENT", "YM"], how="outer")
        .merge(iron, on=["PATIENT", "YM"], how="outer")
        .merge(tibc, on=["PATIENT", "YM"], how="outer")
        .merge(sat, on=["PATIENT", "YM"], how="outer")
        .merge(stool_hgb, on=["PATIENT", "YM"], how="outer")
        .fillna(0)
    )

    feats["gi_signal"] = (feats["gi_condition_count"] > 0).astype(int)
    feats["repeat_enc_signal"] = (feats["encounter_count"] >= 2).astype(int)
    feats["anemia_signal"] = ((feats["hgb_min"] > 0) & (feats["hgb_min"] < args.hgb_low_threshold)).astype(int)
    feats["signals_sum"] = feats["gi_signal"] + feats["repeat_enc_signal"] + feats["anemia_signal"]
    feats["escalation_positive"] = (feats["signals_sum"] >= 2).astype(int)

    feats = feats.set_index(["PATIENT", "YM"])

    rows = []
    lo = 0 if args.include_anchor_month else 1

    for _, a in anchors.iterrows():
        pid = a["PATIENT"]
        anchor_ym = int(a["ANCHOR_YM"])
        is_case = int(a["outcome_has_crc"]) == 1

        if pid in patients.index:
            bd = patients.at[pid, "BIRTHDATE"]
            is_male = int(patients.at[pid, "is_male"])
        else:
            bd = pd.NaT
            is_male = 0

        for i in range(args.lookback_months + 1):
            ym = anchor_ym - i
            year = ym // 12
            month = ym % 12 or 12
            if month == 12:
                year -= 1
            m_start = pd.Timestamp(year=year, month=month, day=1)

            age_years = None
            if pd.notna(bd):
                age_years = (m_start - bd).days / 365.25

            if (pid, ym) in feats.index:
                r = feats.loc[(pid, ym)]
            else:
                r = pd.Series(dtype=float)

            rows.append({
                "PATIENT": pid,
                "COHORT": a["COHORT"],
                "outcome_has_crc": a["outcome_has_crc"],
                "ANCHOR_DATE": a["ANCHOR_DATE"].date(),
                "MONTH_INDEX_BACK": i,
                "MONTH_START": m_start.date(),
                "months_to_crc": i,
                "age_years": age_years,
                "is_male": is_male,

                "encounter_count": int(r.get("encounter_count", 0)),
                "gi_condition_count": int(r.get("gi_condition_count", 0)),
                "colonoscopy_count": int(r.get("colonoscopy_count", 0)),
                "occult_blood_count": int(r.get("occult_blood_count", 0)),

                "hgb_count": int(r.get("hgb_count", 0)),
                "hgb_min": float(r.get("hgb_min", 0)) if r.get("hgb_min", 0) != 0 else None,
                "hct_count": int(r.get("hct_count", 0)),
                "hct_min": float(r.get("hct_min", 0)) if r.get("hct_min", 0) != 0 else None,
                "mcv_count": int(r.get("mcv_count", 0)),
                "mcv_min": float(r.get("mcv_min", 0)) if r.get("mcv_min", 0) != 0 else None,
                "ferritin_count": int(r.get("ferritin_count", 0)),
                "ferritin_min": float(r.get("ferritin_min", 0)) if r.get("ferritin_min", 0) != 0 else None,
                "iron_count": int(r.get("iron_count", 0)),
                "iron_min": float(r.get("iron_min", 0)) if r.get("iron_min", 0) != 0 else None,
                "tibc_count": int(r.get("tibc_count", 0)),
                "tibc_min": float(r.get("tibc_min", 0)) if r.get("tibc_min", 0) != 0 else None,
                "iron_sat_count": int(r.get("iron_sat_count", 0)),
                "iron_sat_min": float(r.get("iron_sat_min", 0)) if r.get("iron_sat_min", 0) != 0 else None,
                "stool_hgb_count": int(r.get("stool_hgb_count", 0)),
                "stool_hgb_min": float(r.get("stool_hgb_min", 0)) if r.get("stool_hgb_min", 0) != 0 else None,

                "gi_signal": int(r.get("gi_signal", 0)),
                "repeat_enc_signal": int(r.get("repeat_enc_signal", 0)),
                "anemia_signal": int(r.get("anemia_signal", 0)),
                "signals_sum": int(r.get("signals_sum", 0)),
                "escalation_positive": int(r.get("escalation_positive", 0)),

                "label_future_crc": int(is_case and lo <= i <= args.label_horizon_months),
            })

    df = pd.DataFrame(rows).sort_values(["PATIENT", "MONTH_INDEX_BACK"])

    df["gi_persist_3m"] = (
        df.groupby("PATIENT")["gi_signal"]
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["gi_count_3m"] = (
        df.groupby("PATIENT")["gi_condition_count"]
        .rolling(3, min_periods=1)
        .sum()
        .reset_index(level=0, drop=True)
    )

    df["encounter_delta"] = df.groupby("PATIENT")["encounter_count"].diff().fillna(0)
    df["encounter_increasing"] = (df["encounter_delta"] > 0).astype(int)

    df["hgb_delta"] = df.groupby("PATIENT")["hgb_min"].diff()
    df["hgb_dropping"] = (df["hgb_delta"] < 0).astype(int)
    df["hgb_min_3m"] = (
        df.groupby("PATIENT")["hgb_min"]
        .rolling(3, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    df["ferritin_min_3m"] = (
        df.groupby("PATIENT")["ferritin_min"]
        .rolling(3, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    df["iron_min_3m"] = (
        df.groupby("PATIENT")["iron_min"]
        .rolling(3, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    df["mcv_min_3m"] = (
        df.groupby("PATIENT")["mcv_min"]
        .rolling(3, min_periods=1)
        .min()
        .reset_index(level=0, drop=True)
    )

    df["colonoscopy_3m"] = (
        df.groupby("PATIENT")["colonoscopy_count"]
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["occult_blood_3m"] = (
        df.groupby("PATIENT")["occult_blood_count"]
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    df["stool_hgb_3m"] = (
        df.groupby("PATIENT")["stool_hgb_count"]
        .rolling(3, min_periods=1)
        .max()
        .reset_index(level=0, drop=True)
    )

    out_path = os.path.join(args.out_dir, f"crc_ml_table_h{args.label_horizon_months}.csv")
    df.to_csv(out_path, index=False)

    print(out_path)
    print(df["label_future_crc"].value_counts())
    print(df.groupby("COHORT")["label_future_crc"].sum())

if __name__ == "__main__":
    main()
