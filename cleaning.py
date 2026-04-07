"""
cleaning.py
This file will load the dataset and clean based off the columns of interest we have.
"""

import os
import pandas as pd
import numpy as np


def load_clean_data(filepath=None):
    """
    Loads and cleans the COVID Future Wave 1B survey data.
    Returns a cleaned pandas DataFrame ready for visualization.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV file. Defaults to the same directory as this
        script, so it works regardless of where your terminal is running from.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with only the columns needed for the 5 visualizations.
    """

    # ── 1. LOAD ───────────────────────────────────────────────────────────────
    if filepath is None:
        filepath = os.path.join(
            os.path.dirname(__file__), "covid_pooled_public_w1b_1.0.0.csv"
        )
    df = pd.read_csv(filepath, low_memory=False)

    # ── 2. SELECT COLUMNS OF INTEREST ────────────────────────────────────────
    # Only keep columns used across the 5 visualizations
    cols_needed = [
        "resp_id",
        # Demographics
        "age", "gender", "educ", "hhincome",
        # Employment context
        "empl_pre", "empl_now", "jobcat_pre", "jobcat_now",
        # Work-from-home: pre, now, expected (Vis 1 + Vis 4)
        "wfh_pre", "wfh_pre_freq", "wfh_now", "wfh_now_days",
        "wfh_expect", "wfh_freq_exp",
        # Commute pre-COVID (Vis 4)
        "pre_work_com_days", "pre_work_pri_mode",
        "pre_work_pri_time", "pre_work_com_dist",
        # Productivity (Vis 3)
        "prod_change",
        "prod_decr_reason_4",   # More distractions at home
        "prod_decr_reason_5",   # Difficult to communicate with co-workers
        "prod_decr_reason_9",   # Too many concerns on mind
        "prod_incr_reason_1",   # Less distractions at home
        "prod_incr_reason_3",   # No commuting time
        # Transportation modes — pre, now, expected (Vis 2 + Vis 5)
        "tr_freq_pre_privateveh", "tr_freq_pre_ridehail", "tr_freq_pre_transit",
        "tr_freq_pre_bikepers",   "tr_freq_pre_bikeshared", "tr_freq_pre_walk",
        "tr_freq_now_privateveh", "tr_freq_now_ridehail",   "tr_freq_now_transit",
        "tr_freq_now_bikepers",   "tr_freq_now_bikeshared", "tr_freq_now_walk",
        "tr_freq_exp_privateveh", "tr_freq_exp_ridehail",   "tr_freq_exp_transit",
        "tr_freq_exp_bikepers",   "tr_freq_exp_bikeshared", "tr_freq_exp_walk",
        # Reasons for transit drop / bike increase (tooltip context)
        "decrease_transit_1", "decrease_transit_2", "decrease_transit_3",
        "increase_bike_1",    "increase_bike_2",    "increase_bike_3",
        # Vehicle/bike ownership
        "driver", "bike",
        # Survey weight
        "weight",
    ]

    df = df[cols_needed].copy()

    # ── 3. REPLACE NON-ANSWER PLACEHOLDERS WITH NaN ───────────────────────────
    # These strings appear when a survey question was skipped due to routing logic;
    # they are not real responses and should be treated as missing.
    non_answers = [
        "Question not displayed to respondent",
        "Seen but unanswered",
    ]
    df.replace(non_answers, np.nan, inplace=True)

    # ── 4. DEMOGRAPHICS ───────────────────────────────────────────────────────

    # Age: coerce to numeric, then bin into readable groups
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 29, 44, 59, 74, 120],
        labels=["18–29", "30–44", "45–59", "60–74", "75+"]
    )

    # Income: make an ordered categorical (preserves sort order in charts)
    income_order = [
        "Less than $15,000",
        "$15,000 to $24,999",
        "$25,000 to $34,999",
        "$35,000 to $49,999",
        "$50,000 to $74,999",
        "$75,000 to $99,999",
        "$100,000 to $124,999",
        "$125,000 to $149,999",
        "$150,000 to $199,999",
        "$200,000 or more",
    ]
    # Filter to known categories first to avoid FutureWarning in newer pandas
    df["hhincome"] = df["hhincome"].where(df["hhincome"].isin(income_order), other=np.nan)
    df["hhincome"] = pd.Categorical(df["hhincome"], categories=income_order, ordered=True)

    # Simplified income brackets for grouped bar charts (Vis 1)
    income_bracket_map = {
        "Less than $15,000":      "Under $35k",
        "$15,000 to $24,999":     "Under $35k",
        "$25,000 to $34,999":     "Under $35k",
        "$35,000 to $49,999":     "$35k–$74k",
        "$50,000 to $74,999":     "$35k–$74k",
        "$75,000 to $99,999":     "$75k–$124k",
        "$100,000 to $124,999":   "$75k–$124k",
        "$125,000 to $149,999":   "$125k+",
        "$150,000 to $199,999":   "$125k+",
        "$200,000 or more":       "$125k+",
    }
    df["income_bracket"] = df["hhincome"].map(income_bracket_map)

    # Education: collapse into 4 tiers for cleaner axis labels
    educ_map = {
        "Some grade/high school":                           "High school or less",
        "Completed high school or GED":                     "High school or less",
        "Some college or technical school":                 "Some college",
        "Bachelor's degree(s) or some graduate school":    "Bachelor's",
        "Completed graduate degree(s)":                     "Graduate degree",
    }
    df["educ_simple"] = df["educ"].map(educ_map)

    # ── 5. WORK-FROM-HOME COLUMNS ─────────────────────────────────────────────
    # NOTE: wfh_pre_freq, wfh_now_days, and wfh_freq_exp will have high null
    # counts (~70–75%) because those questions were only shown to workers who
    # had WFH access. This is expected survey routing, NOT a data quality issue.

    # wfh_pre_freq: ordinal string → approximate days/week equivalent
    freq_to_num = {
        "Never":              0.0,
        "A few times/year":   0.1,
        "A few times/month":  0.5,
        "Once/week":          1.0,
        "A few times/week":   3.0,
        "Every day":          5.0,
    }
    df["wfh_pre_freq_num"] = df["wfh_pre_freq"].map(freq_to_num)

    # wfh_now_days: numeric days in past 7 days — just coerce type
    df["wfh_now_days"] = pd.to_numeric(df["wfh_now_days"], errors="coerce")

    # wfh_freq_exp: expected post-COVID frequency → same numeric scale
    df["wfh_freq_exp_num"] = df["wfh_freq_exp"].map(freq_to_num)

    # ── 6. TRANSPORTATION MODE COLUMNS ───────────────────────────────────────
    # The three time periods use different encodings; we standardize each below.

    transport_modes = ["privateveh", "ridehail", "transit", "bikepers", "bikeshared", "walk"]

    # PRE-COVID: ordinal strings → numeric frequency (same scale as WFH above)
    for mode in transport_modes:
        col = f"tr_freq_pre_{mode}"
        df[col] = df[col].map(freq_to_num)

    # DURING COVID (past 7 days): already numeric days 0–7, just coerce
    for mode in transport_modes:
        col = f"tr_freq_now_{mode}"
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # EXPECTED POST-COVID: relative change strings → -2 to +2 ordinal score
    exp_change_map = {
        "Much less than before":     -2,
        "Somewhat less than before": -1,
        "About the same":             0,
        "Somewhat more than before":  1,
        "Much more than before":      2,
    }
    for mode in transport_modes:
        col = f"tr_freq_exp_{mode}"
        df[col] = df[col].map(exp_change_map)

    # ── 7. PRODUCTIVITY COLUMN ────────────────────────────────────────────────
    # NOTE: ~45% null because question was only shown to workers.
    prod_map = {
        "Decreased significantly":                                              -2,
        "Decreased somewhat":                                                   -1,
        "About the same":                                                        0,
        "In some ways it has increased and in other ways it has decreased":      0,
        "Increased somewhat":                                                    1,
        "Increased significantly":                                               2,
    }
    df["prod_change_num"] = df["prod_change"].map(prod_map)

    # Simplify job category labels for axis readability (Vis 3)
    jobcat_map = {
        "Professional, managerial, or technical": "Professional/Manager",
        "Sales or service":                        "Sales/Service",
        "Clerical or administrative support":      "Clerical/Admin",
        "Skilled crafts or trades":                "Skilled Trades",
        "Farming, fishing, or forestry":           "Farming/Forestry",
        "Production or manufacturing":             "Production/Mfg",
        "Other":                                   "Other",
    }
    df["jobcat_pre_simple"] = df["jobcat_pre"].map(jobcat_map)

    # ── 8. COMMUTE COLUMNS ────────────────────────────────────────────────────
    # NOTE: ~50% null because only shown to workers who commuted pre-COVID.
    df["pre_work_pri_time"] = pd.to_numeric(df["pre_work_pri_time"], errors="coerce")
    df["pre_work_com_days"] = pd.to_numeric(df["pre_work_com_days"], errors="coerce")
    df["pre_work_com_dist"] = pd.to_numeric(df["pre_work_com_dist"], errors="coerce")

    # ── 9. DROP ROWS MISSING CORE DEMOGRAPHICS ───────────────────────────────
    # Only drop rows where all three core demographic fields are missing.
    # Visualization-specific nulls (WFH, commute, productivity) are kept
    # because each viz will filter to its own relevant subset.
    df = df.dropna(subset=["age", "hhincome", "educ"])
    df = df.reset_index(drop=True)

    return df


# ── QUICK SANITY CHECK (runs only when executed directly) ────────────────────
if __name__ == "__main__":
    df = load_clean_data()
    print(f"Shape: {df.shape}")
    print(f"\nColumns ({len(df.columns)}):\n{list(df.columns)}")
    print(f"\nIncome bracket counts:\n{df['income_bracket'].value_counts()}")
    print(f"\nAge group counts:\n{df['age_group'].value_counts()}")
    print(f"\nKey null counts:")
    key_cols = [
        "income_bracket", "age_group", "educ_simple",
        "wfh_pre_freq_num", "wfh_now_days", "wfh_freq_exp_num",
        "prod_change_num", "jobcat_pre_simple",
        "tr_freq_pre_transit", "tr_freq_now_transit", "tr_freq_exp_transit",
        "pre_work_pri_time",
    ]
    for col in key_cols:
        n_null = df[col].isna().sum()
        pct = n_null / len(df) * 100
        print(f"  {col}: {n_null} nulls ({pct:.1f}%)")