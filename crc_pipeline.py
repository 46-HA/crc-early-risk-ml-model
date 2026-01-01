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

# Labs (observations)
HGB_DESC_REGEX_DEFAULT = r"(?:\bhemoglobin\b|\bhgb\b)"
HCT_DESC_REGEX_DEFAULT = r"(?:\bhematocrit\b|\bhct\b)"
MCV_DESC_REGEX_DEFAULT = r"(?:\bmcv\b|mean corpuscular volume)"
FERRITIN_DESC_REGEX_DEFAULT = r"(?:\bferritin\b)"
IRON_DESC_REGEX_DEFAULT = r"(?:\biron\b|\bserum iron\b)"
STOOL_HGB_DESC_REGEX_DEFAULT = r"(?:stool hemoglobin|fecal hemoglobin|fecal occult blood|occult blood|fit\b|guaiac)"

# Option-3 signals
FAM_HX_REGEX_DEFAULT = r"(?:family history).*(?:colon|colorectal|rectal).*?(?:cancer|malignan|neoplasm)"

# Procedures (screening)
COLONOSCOPY_REGEX_DEFAULT = r"(?:colonoscopy|colonoscopy biopsy|flexible sigmoidoscopy|sigmoidoscopy)"
FOBT_FIT_REGEX_DEFAULT = r"(?:fecal occult blood|occult blood|fit\b|guaiac)"

# Imaging studies (proxy)
CT_ABD_PELVIS_REGEX_DEFAULT = r"(?:ct).*(?:abdomen|abdominal|pelvis|pelvic)|(?:abdomen|pelvis).*(?:ct)"
GI_IMAGING_REGEX_DEFAULT = r"(?:colon|colonic|colorectal|rectal).*(?:ct|mri|imaging|scan)|(?:ct|mri).*(?:colon|colonic|colorectal|rectal)"

def parse_data_dirs(args):
    if args.data_dirs and args.data_dirs.strip():
        dirs = [x.strip() for x in args.data_dirs.split(",") if x.strip()]
    else:
        dirs = [args.data_dir]
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
    print(f"loaded {filename}: {len(out):,} rows from {len(used)} files", flush=True)
    return out

def pick_date_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"missing date col, tried {candidates}, found {list(df.columns)}")

def pick_text_col(df, candidates, required=False, label=""):
    """
    Pick first available text column from candidates.
    If required=False and none found, returns None.
    """
    for c in candidates:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"{label} missing text col; tried {candidates}; found {list(df.columns)}")
    return None

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

def build_pid_index(df, pid_col="PATIENT"):
    if df is None or len(df) == 0:
        return {}
    return {pid: g for pid, g in df.groupby(pid_col, sort=False)}

def obs_filter(observations, obs_date, text_col, desc_regex):
    if text_col is None:
        return observations.iloc[0:0].copy()
    mask = observations[text_col].fillna("").str.contains(desc_regex, flags=re.IGNORECASE, regex=True)
    out = observations.loc[mask].dropna(subset=[obs_date]).copy()
    out["VALUE_NUM"] = pd.to_numeric(out.get("VALUE"), errors="coerce")
    return out

def month_stats_numeric(obs_df, date_col, m_start, m_end):
    if obs_df is None or len(obs_df) == 0:
        return 0, None, None
    x = obs_df[(obs_df[date_col] >= m_start) & (obs_df[date_col] < m_end)]["VALUE_NUM"].dropna()
    if x.empty:
        return 0, None, None
    return int(x.shape[0]), float(x.min()), float(x.mean())

def month_any(df, date_col, m_start, m_end):
    if df is None or len(df) == 0:
        return 0, 0
    sub = df[(df[date_col] >= m_start) & (df[date_col] < m_end)]
    return int(len(sub) > 0), int(len(sub))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_all")
    ap.add_argument("--data_dirs", default="", help="comma-separated, e.g. data_10k,data_30k")
    ap.add_argument("--out_dir", default="output")
    ap.add_argument("--anchors_csv", default=os.path.join("output", "anchors_crc_all.csv"))
    ap.add_argument("--lookback_months", type=int, default=24)
    ap.add_argument("--hgb_low_threshold", type=float, default=11.0)

    ap.add_argument("--gi_regex", default=GI_DESC_REGEX_DEFAULT)

    ap.add_argument("--hgb_regex", default=HGB_DESC_REGEX_DEFAULT)
    ap.add_argument("--hct_regex", default=HCT_DESC_REGEX_DEFAULT)
    ap.add_argument("--mcv_regex", default=MCV_DESC_REGEX_DEFAULT)
    ap.add_argument("--ferritin_regex", default=FERRITIN_DESC_REGEX_DEFAULT)
    ap.add_argument("--iron_regex", default=IRON_DESC_REGEX_DEFAULT)
    ap.add_argument("--stool_hgb_regex", default=STOOL_HGB_DESC_REGEX_DEFAULT)

    ap.add_argument("--fam_hx_regex", default=FAM_HX_REGEX_DEFAULT)
    ap.add_argument("--colonoscopy_regex", default=COLONOSCOPY_REGEX_DEFAULT)
    ap.add_argument("--fobt_fit_regex", default=FOBT_FIT_REGEX_DEFAULT)
    ap.add_argument("--ct_abd_pelvis_regex", default=CT_ABD_PELVIS_REGEX_DEFAULT)
    ap.add_argument("--gi_imaging_regex", default=GI_IMAGING_REGEX_DEFAULT)

    ap.add_argument("--progress_every", type=int, default=200)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if not os.path.exists(args.anchors_csv):
        raise FileNotFoundError(
            f"Missing anchors file: {args.anchors_csv}\n"
            f"Run: python3 build_anchors_crc.py --lookback_months {args.lookback_months}"
        )

    anchors_all = pd.read_csv(args.anchors_csv)
    if "PATIENT" not in anchors_all.columns or "ANCHOR_DATE" not in anchors_all.columns:
        raise KeyError(f"anchors_csv must contain PATIENT, ANCHOR_DATE. Found: {list(anchors_all.columns)}")

    anchors_all["ANCHOR_DATE"] = pd.to_datetime(anchors_all["ANCHOR_DATE"], errors="coerce")
    anchors_all = anchors_all.dropna(subset=["ANCHOR_DATE"]).copy()

    have_cohort = "COHORT" in anchors_all.columns
    have_outcome = "outcome_has_crc" in anchors_all.columns

    data_dirs = parse_data_dirs(args)
    print("DATA_DIRS:", data_dirs, flush=True)

    # Load needed files
    conditions = read_concat_csv(data_dirs, "conditions.csv")
    encounters = read_concat_csv(data_dirs, "encounters.csv")
    observations = read_concat_csv(data_dirs, "observations.csv")
    procedures = read_concat_csv(data_dirs, "procedures.csv")
    imaging = read_concat_csv(data_dirs, "imaging_studies.csv")

    # Date columns
    cond_date = pick_date_col(conditions, ["START", "DATE", "STOP"])
    enc_date  = pick_date_col(encounters, ["START", "DATE", "STOP"])
    obs_date  = pick_date_col(observations, ["DATE", "START", "STOP"])
    proc_date = pick_date_col(procedures, ["DATE", "START", "STOP"])
    img_date  = pick_date_col(imaging, ["DATE", "START", "STOP"])

    # Text columns (some Synthea versions differ!)
    cond_text = pick_text_col(conditions, ["DESCRIPTION", "REASONDESCRIPTION", "CODE", "DISPLAY"], required=True, label="conditions")
    obs_text  = pick_text_col(observations, ["DESCRIPTION", "REASONDESCRIPTION", "TYPE", "CODE", "DISPLAY"], required=True, label="observations")
    proc_text = pick_text_col(procedures, ["DESCRIPTION", "REASONDESCRIPTION", "CODE", "DISPLAY"], required=False, label="procedures")
    img_text  = pick_text_col(imaging, [
        "DESCRIPTION",
        "BODYSITE_DESCRIPTION",
        "MODALITY_DESCRIPTION",
        "PROCEDURE_DESCRIPTION",
        "BODYSITE",
        "MODALITY",
        "PROCEDURE",
        "CODE",
        "DISPLAY",
    ], required=False, label="imaging_studies")

    if proc_text is None:
        # Most Synthea versions do have DESCRIPTION; if not, we can’t regex procedures.
        print("warning: procedures has no usable text column; screening features disabled", flush=True)

    if img_text is None:
        print("warning: imaging_studies has no usable text column; imaging proxy features disabled", flush=True)

    # Reduce columns early
    conditions = conditions[[c for c in ["PATIENT", cond_text, cond_date] if c in conditions.columns]].copy()
    conditions.rename(columns={cond_text: "TEXT"}, inplace=True)

    encounters = encounters[[c for c in ["PATIENT", enc_date] if c in encounters.columns]].copy()

    observations = observations[[c for c in ["PATIENT", obs_text, obs_date, "VALUE"] if c in observations.columns]].copy()
    observations.rename(columns={obs_text: "TEXT"}, inplace=True)

    procedures = procedures[[c for c in ["PATIENT", proc_text, proc_date] if c in procedures.columns and proc_text is not None]].copy()
    if proc_text is not None:
        procedures.rename(columns={proc_text: "TEXT"}, inplace=True)

    imaging = imaging[[c for c in ["PATIENT", img_text, img_date] if c in imaging.columns and img_text is not None]].copy()
    if img_text is not None:
        imaging.rename(columns={img_text: "TEXT"}, inplace=True)

    # Datetimes
    conditions[cond_date] = to_dt_naive(conditions[cond_date])
    encounters[enc_date]  = to_dt_naive(encounters[enc_date])
    observations[obs_date]= to_dt_naive(observations[obs_date])
    if len(procedures) > 0:
        procedures[proc_date] = to_dt_naive(procedures[proc_date])
    if len(imaging) > 0:
        imaging[img_date] = to_dt_naive(imaging[img_date])

    # GI conditions
    gi_mask = conditions["TEXT"].fillna("").str.contains(args.gi_regex, flags=re.IGNORECASE, regex=True)
    gi_conds = conditions.loc[gi_mask].dropna(subset=[cond_date]).copy()

    # Family history conditions
    fam_mask = conditions["TEXT"].fillna("").str.contains(args.fam_hx_regex, flags=re.IGNORECASE, regex=True)
    fam_conds = conditions.loc[fam_mask].dropna(subset=[cond_date]).copy()

    # Lab observations
    hgb_obs = obs_filter(observations, obs_date, "TEXT", args.hgb_regex)
    hct_obs = obs_filter(observations, obs_date, "TEXT", args.hct_regex)
    mcv_obs = obs_filter(observations, obs_date, "TEXT", args.mcv_regex)
    ferr_obs = obs_filter(observations, obs_date, "TEXT", args.ferritin_regex)
    iron_obs = obs_filter(observations, obs_date, "TEXT", args.iron_regex)

    stool_obs = observations[
        observations["TEXT"].fillna("").str.contains(args.stool_hgb_regex, flags=re.IGNORECASE, regex=True)
    ].dropna(subset=[obs_date]).copy()

    # Screening procedures (optional)
    colonoscopy_proc = procedures.iloc[0:0].copy()
    fobt_proc = procedures.iloc[0:0].copy()
    if len(procedures) > 0:
        colonoscopy_mask = procedures["TEXT"].fillna("").str.contains(args.colonoscopy_regex, flags=re.IGNORECASE, regex=True)
        fobt_mask = procedures["TEXT"].fillna("").str.contains(args.fobt_fit_regex, flags=re.IGNORECASE, regex=True)
        colonoscopy_proc = procedures.loc[colonoscopy_mask].dropna(subset=[proc_date]).copy()
        fobt_proc = procedures.loc[fobt_mask].dropna(subset=[proc_date]).copy()

    # Imaging proxies (optional)
    ct_img = imaging.iloc[0:0].copy()
    gi_img = imaging.iloc[0:0].copy()
    if len(imaging) > 0:
        ct_mask = imaging["TEXT"].fillna("").str.contains(args.ct_abd_pelvis_regex, flags=re.IGNORECASE, regex=True)
        gi_img_mask = imaging["TEXT"].fillna("").str.contains(args.gi_imaging_regex, flags=re.IGNORECASE, regex=True)
        ct_img = imaging.loc[ct_mask].dropna(subset=[img_date]).copy()
        gi_img = imaging.loc[gi_img_mask].dropna(subset=[img_date]).copy()

    print(f"gi_conds rows: {len(gi_conds):,}", flush=True)
    print(f"fam_conds rows: {len(fam_conds):,}", flush=True)
    print(f"hgb_obs rows: {len(hgb_obs):,}", flush=True)
    print(f"hct_obs rows: {len(hct_obs):,}", flush=True)
    print(f"mcv_obs rows: {len(mcv_obs):,}", flush=True)
    print(f"ferritin_obs rows: {len(ferr_obs):,}", flush=True)
    print(f"iron_obs rows: {len(iron_obs):,}", flush=True)
    print(f"stool_obs rows: {len(stool_obs):,}", flush=True)
    print(f"colonoscopy_proc rows: {len(colonoscopy_proc):,}", flush=True)
    print(f"fobt_proc rows: {len(fobt_proc):,}", flush=True)
    print(f"ct_img rows: {len(ct_img):,}", flush=True)
    print(f"gi_img rows: {len(gi_img):,}", flush=True)

    # Index by PATIENT
    print("indexing by PATIENT...", flush=True)
    gi_by_pid    = build_pid_index(gi_conds, "PATIENT")
    fam_by_pid   = build_pid_index(fam_conds, "PATIENT")
    enc_by_pid   = build_pid_index(encounters.dropna(subset=[enc_date]), "PATIENT")
    hgb_by_pid   = build_pid_index(hgb_obs, "PATIENT")
    hct_by_pid   = build_pid_index(hct_obs, "PATIENT")
    mcv_by_pid   = build_pid_index(mcv_obs, "PATIENT")
    ferr_by_pid  = build_pid_index(ferr_obs, "PATIENT")
    iron_by_pid  = build_pid_index(iron_obs, "PATIENT")
    stool_by_pid = build_pid_index(stool_obs, "PATIENT")
    col_by_pid   = build_pid_index(colonoscopy_proc, "PATIENT")
    fobt_by_pid  = build_pid_index(fobt_proc, "PATIENT")
    ct_by_pid    = build_pid_index(ct_img, "PATIENT")
    giimg_by_pid = build_pid_index(gi_img, "PATIENT")
    print("done indexing.", flush=True)

    EMPTY_GI    = gi_conds.iloc[0:0]
    EMPTY_FAM   = fam_conds.iloc[0:0]
    EMPTY_ENC   = encounters.iloc[0:0]
    EMPTY_HGB   = hgb_obs.iloc[0:0]
    EMPTY_HCT   = hct_obs.iloc[0:0]
    EMPTY_MCV   = mcv_obs.iloc[0:0]
    EMPTY_FERR  = ferr_obs.iloc[0:0]
    EMPTY_IRON  = iron_obs.iloc[0:0]
    EMPTY_STOOL = stool_obs.iloc[0:0]
    EMPTY_COL   = colonoscopy_proc.iloc[0:0]
    EMPTY_FOBT  = fobt_proc.iloc[0:0]
    EMPTY_CT    = ct_img.iloc[0:0]
    EMPTY_GIIMG = gi_img.iloc[0:0]

    monthly_rows = []
    window_rows = []

    n = len(anchors_all)
    for j, (_, ar) in enumerate(anchors_all.iterrows(), 1):
        if args.progress_every > 0 and (j % args.progress_every == 0 or j == 1 or j == n):
            print(f"processed {j}/{n} anchors...", flush=True)

        pid = ar["PATIENT"]
        anchor = ar["ANCHOR_DATE"]
        if pd.isna(anchor):
            continue

        cohort = ar["COHORT"] if have_cohort else None
        outcome = int(ar["outcome_has_crc"]) if have_outcome and pd.notna(ar["outcome_has_crc"]) else 0
        is_case = outcome == 1

        gi_p    = gi_by_pid.get(pid, EMPTY_GI)
        fam_p   = fam_by_pid.get(pid, EMPTY_FAM)
        enc_p   = enc_by_pid.get(pid, EMPTY_ENC)
        hgb_p   = hgb_by_pid.get(pid, EMPTY_HGB)
        hct_p   = hct_by_pid.get(pid, EMPTY_HCT)
        mcv_p   = mcv_by_pid.get(pid, EMPTY_MCV)
        ferr_p  = ferr_by_pid.get(pid, EMPTY_FERR)
        iron_p  = iron_by_pid.get(pid, EMPTY_IRON)
        stool_p = stool_by_pid.get(pid, EMPTY_STOOL)
        col_p   = col_by_pid.get(pid, EMPTY_COL)
        fobt_p  = fobt_by_pid.get(pid, EMPTY_FOBT)
        ct_p    = ct_by_pid.get(pid, EMPTY_CT)
        giimg_p = giimg_by_pid.get(pid, EMPTY_GIIMG)

        feats_newest_first = []

        for i, m_start, m_end in iter_month_bins(anchor, args.lookback_months):
            gi_pid  = gi_p[(gi_p[cond_date] >= m_start) & (gi_p[cond_date] < m_end)]
            enc_pid = enc_p[(enc_p[enc_date]  >= m_start) & (enc_p[enc_date]  < m_end)]
            hgb_month = hgb_p[(hgb_p[obs_date] >= m_start) & (hgb_p[obs_date] < m_end)]

            gi_signal = int(len(gi_pid) > 0)
            repeat_enc_signal = int(len(enc_pid) >= 2)
            anemia_signal = int((hgb_month["VALUE_NUM"].dropna() < args.hgb_low_threshold).any())

            # Family history cumulative before month end
            fam_before = fam_p[fam_p[cond_date] < m_end]
            family_history_crc = int(len(fam_before) > 0)
            fam_hx_count = int(len(fam_before))

            # Screening cumulative before month end
            col_before = col_p[col_p.get(proc_date, pd.Series(dtype="datetime64[ns]")) < m_end] if len(col_p) else col_p
            fobt_before = fobt_p[fobt_p.get(proc_date, pd.Series(dtype="datetime64[ns]")) < m_end] if len(fobt_p) else fobt_p
            had_colonoscopy = int(len(col_before) > 0)
            had_fobt_fit = int(len(fobt_before) > 0)
            screening_any = int(had_colonoscopy or had_fobt_fit)
            colonoscopy_count = int(len(col_before))
            fobt_fit_count = int(len(fobt_before))

            # Imaging monthly counts
            ct_any, ct_count = month_any(ct_p, img_date, m_start, m_end)
            giimg_any, giimg_count = month_any(giimg_p, img_date, m_start, m_end)

            signals_sum = gi_signal + anemia_signal + repeat_enc_signal
            escalation_positive = int(signals_sum >= 2)

            hgb_cnt, hgb_min, hgb_mean = month_stats_numeric(hgb_p,  obs_date, m_start, m_end)
            hct_cnt, hct_min, hct_mean = month_stats_numeric(hct_p,  obs_date, m_start, m_end)
            mcv_cnt, mcv_min, mcv_mean = month_stats_numeric(mcv_p,  obs_date, m_start, m_end)
            fer_cnt, fer_min, fer_mean = month_stats_numeric(ferr_p, obs_date, m_start, m_end)
            iron_cnt, iron_min, iron_mean = month_stats_numeric(iron_p, obs_date, m_start, m_end)
            stool_any, stool_cnt = month_any(stool_p, obs_date, m_start, m_end)

            row = {
                "PATIENT": pid,
                "ANCHOR_DATE": anchor.date() if hasattr(anchor, "date") else anchor,
                "MONTH_INDEX_BACK": i,
                "MONTH_START": m_start.date(),
                "MONTH_END": m_end.date(),

                "anemia_signal": anemia_signal,
                "gi_signal": gi_signal,
                "repeat_enc_signal": repeat_enc_signal,
                "gi_condition_count": int(len(gi_pid)),
                "encounter_count": int(len(enc_pid)),

                "hgb_obs_count": hgb_cnt,
                "hgb_min_value": hgb_min,
                "hgb_mean_value": hgb_mean,
                "hct_obs_count": hct_cnt,
                "hct_min_value": hct_min,
                "hct_mean_value": hct_mean,
                "mcv_obs_count": mcv_cnt,
                "mcv_min_value": mcv_min,
                "mcv_mean_value": mcv_mean,
                "ferritin_obs_count": fer_cnt,
                "ferritin_min_value": fer_min,
                "ferritin_mean_value": fer_mean,
                "iron_obs_count": iron_cnt,
                "iron_min_value": iron_min,
                "iron_mean_value": iron_mean,
                "stool_hgb_any": stool_any,
                "stool_hgb_count": stool_cnt,

                "family_history_crc": family_history_crc,
                "fam_hx_count": fam_hx_count,
                "had_colonoscopy": had_colonoscopy,
                "had_fobt_fit": had_fobt_fit,
                "screening_any": screening_any,
                "colonoscopy_count": colonoscopy_count,
                "fobt_fit_count": fobt_fit_count,
                "ct_abd_pelvis_any": ct_any,
                "ct_abd_pelvis_count": ct_count,
                "gi_imaging_any": giimg_any,
                "gi_imaging_count": giimg_count,

                "signals_sum": signals_sum,
                "escalation_positive": escalation_positive,
            }

            if have_cohort:
                row["COHORT"] = cohort
            if have_outcome:
                row["outcome_has_crc"] = outcome

            feats_newest_first.append(row)

        monthly_rows.extend(feats_newest_first)

        if not is_case:
            window_rows.append({"PATIENT": pid, "ANCHOR_DATE": anchor, "window_found": 0,
                                "window_month1_start": None, "window_month2_start": None,
                                "window_month1_index_back": None, "window_month2_index_back": None})
        else:
            feats_oldest_first = list(reversed(feats_newest_first))
            flags = [bool(x["escalation_positive"]) for x in feats_oldest_first]
            idx = first_two_consecutive_true(flags)

            if idx is None:
                window_rows.append({"PATIENT": pid, "ANCHOR_DATE": anchor, "window_found": 0,
                                    "window_month1_start": None, "window_month2_start": None,
                                    "window_month1_index_back": None, "window_month2_index_back": None})
            else:
                m1 = feats_oldest_first[idx]
                m2 = feats_oldest_first[idx+1]
                window_rows.append({"PATIENT": pid, "ANCHOR_DATE": anchor, "window_found": 1,
                                    "window_month1_start": m1["MONTH_START"], "window_month2_start": m2["MONTH_START"],
                                    "window_month1_index_back": m1["MONTH_INDEX_BACK"], "window_month2_index_back": m2["MONTH_INDEX_BACK"]})

    monthly_df = pd.DataFrame(monthly_rows).sort_values(["PATIENT", "MONTH_INDEX_BACK"])
    windows_df = pd.DataFrame(window_rows)

    monthly_path = os.path.join(args.out_dir, "crc_monthly_features.csv")
    windows_path = os.path.join(args.out_dir, "crc_windows.csv")
    monthly_df.to_csv(monthly_path, index=False)
    windows_df.to_csv(windows_path, index=False)

    print("wrote", monthly_path, f"({len(monthly_df):,} rows)", flush=True)
    print("wrote", windows_path, f"({len(windows_df):,} rows)", flush=True)

    feats = monthly_df.copy()
    wins = windows_df.copy()

    feats["MONTH_START"] = pd.to_datetime(feats["MONTH_START"], errors="coerce")
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

    ml_path = os.path.join(args.out_dir, "crc_ml_table.csv")
    df.to_csv(ml_path, index=False)
    print("wrote", ml_path, f"({len(df):,} rows)", flush=True)

if __name__ == "__main__":
    main()
