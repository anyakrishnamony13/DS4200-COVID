import altair as alt
import pandas as pd
import numpy as np
import webbrowser
import os
from cleaning import load_clean_data


# ── STYLING ───────────────────────────────────────────────────────────────────
FONT = "Georgia"

COLORS = {
    "Increased": "#55A868",
    "Decreased": "#C44E52"
}


# ═════════════════════════════════════════════════════════════════════════════
# VIS 3 — Diverging Bar Chart
# "Productivity Winners and Losers: Who Thrived Working from Home?"
# ═════════════════════════════════════════════════════════════════════════════

def vis3_productivity_diverging(df):
    """
    Productivity Winners and Losers: Who Thrived Working from Home?

    This visualization examines self-reported changes in productivity during
    the shift to remote work, broken down by pre-pandemic job category.

    The chart uses a diverging bar design:
    - Bars extending to the right represent respondents reporting increased productivity
    - Bars extending to the left represent respondents reporting decreased productivity

    Data preprocessing:
    - Free-text survey responses for productivity change are grouped into
      "Increased" and "Decreased" categories for clarity
    - Job categories are used to compare occupational differences

    Key insight:
    Professional and managerial workers are more likely to report productivity gains
    under remote work, while workers in service, sales, and manual labor roles show
    higher rates of decreased productivity. This highlights how the benefits of
    remote work are unevenly distributed across occupational groups.
    """
    
    # --- Clean + recode messy text into 3 buckets ---
    def categorize_change(x):
        x = str(x).lower()
        if "increase" in x:
            return "Increased"
        elif "decrease" in x:
            return "Decreased"
        else:
            return "Same/Mixed"

    data = df.copy()
    data["prod_change_clean"] = data["prod_change"].apply(categorize_change)

    # keep only clear diverging categories
    data = data[
        data["prod_change_clean"].isin(["Increased", "Decreased"]) &
        data["jobcat_pre"].notna()
    ]

    # --- Aggregate ---
    grouped = (
        data
        .groupby(["jobcat_pre", "prod_change_clean"])
        .size()
        .reset_index(name="count")
    )

    totals = grouped.groupby("jobcat_pre")["count"].transform("sum")
    grouped["pct"] = grouped["count"] / totals * 100

    # diverging bars (left = negative)
    grouped["pct"] = grouped.apply(
        lambda r: -r["pct"] if r["prod_change_clean"] == "Decreased" else r["pct"],
        axis=1
    )

    # --- Chart ---
    chart = (
        alt.Chart(grouped)
        .mark_bar()
        .encode(
            y=alt.Y(
                "jobcat_pre:N",
                sort="-x",
                title="Job Category",
                axis=alt.Axis(labelFont=FONT, titleFont=FONT)
            ),
            x=alt.X(
                "pct:Q",
                title="% Reporting Productivity Change",
                axis=alt.Axis(format=".0f", labelFont=FONT, titleFont=FONT)
            ),
            color=alt.Color(
                "prod_change_clean:N",
                scale=alt.Scale(
                    domain=["Increased", "Decreased"],
                    range=[COLORS["Increased"], COLORS["Decreased"]]
                ),
                legend=alt.Legend(title="Productivity Change")
            ),
            tooltip=[
                "jobcat_pre:N",
                "prod_change_clean:N",
                alt.Tooltip("pct:Q", format=".1f")
            ]
        )
        .properties(
            width=600,
            height=400,
            title="Productivity Winners and Losers: Who Thrived Working from Home?"
        )
    )

    return chart


# ═════════════════════════════════════════════════════════════════════════════
# VIS 4 — Interactive Scatter + Linked Bar Chart
# "Commute Time vs. WFH Adoption"
# ═════════════════════════════════════════════════════════════════════════════

def vis4_commute_wfh_interactive(df):
    """
    Commute Time vs. WFH Adoption: Did Long Commuters Embrace Remote Work More?

    This interactive visualization explores how pre-pandemic commute time relates
    to current WFH behavior, with a focus on education level differences.

    Top panel:
    - Scatter plot of commute time vs. current WFH days
    - Color encodes education level
    - Brush selection allows filtering of respondents

    Bottom panel:
    - Bar chart showing expected post-COVID WFH frequency
    - Updates dynamically based on selected points in scatter plot

    Key insight:
    Individuals with longer pre-pandemic commutes tend to work from home more
    frequently and expect to continue doing so, suggesting commute burden plays
    an important role in shaping remote work preferences.
    """

    # ── CLEAN DATA ───────────────────────────────────────────────────────────
    data = df[
        df["pre_work_pri_time"].notna() &
        df["wfh_now_days"].notna() &
        df["wfh_freq_exp"].notna()
    ].copy()

    # ── INTERACTION ─────────────────────────────────────────────────────────
    brush = alt.selection_interval()

    WFH_ORDER = [
        "Never",
        "A few times/year",
        "A few times/month",
        "Once/week",
        "A few times/week",
        "Every day"
    ]
    
    # ── SCATTER PLOT ─────────────────────────────────────────────────────────
    scatter = (
        alt.Chart(data)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X(
                "pre_work_pri_time:Q",
                title="Pre-COVID Commute Time (minutes)"
            ),
            y=alt.Y(
                "wfh_now_days:Q",
                title="WFH Days Per Week (Now)"
            ),
            color=alt.Color(
                "wfh_freq_exp:N",
                title="Expected WFH Frequency",
                scale=alt.Scale(
                    domain=WFH_ORDER
                ),
                legend=alt.Legend(
                    orient="right",
                    labelFont=FONT,
                    titleFont=FONT
                )
            ),
            tooltip=[
                "pre_work_pri_time:Q",
                "wfh_now_days:Q",
                "wfh_freq_exp:N",
            ]
        )
        .add_params(brush)
        .properties(width=600, height=400)
    )

    # ── LINKED BAR CHART ─────────────────────────────────────────────────────
    bars = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X(
                "wfh_freq_exp:N",
                title="Expected Post-COVID WFH Frequency",
                sort=WFH_ORDER
            ),
            y=alt.Y(
                "count()",
                title="Respondents"
            ),
            color=alt.Color(
                "wfh_freq_exp:N",
                legend=None
            )
        )
        .transform_filter(brush)
        .properties(width=600, height=200)
    )

    # ── COMBINE ──────────────────────────────────────────────────────────────
    chart = alt.vconcat(
        scatter,
        bars
    ).properties(
        title=alt.TitleParams(
            text="Commute Time vs. WFH Adoption: Did Long Commuters Embrace Remote Work More?",
            subtitle=[
                "Select regions in the scatter plot to filter expected work from home behavior in the bar chart below."
            ],
            font=FONT,
            fontSize=16,
            subtitleFont=FONT,
            subtitleFontSize=12,
            anchor="start"
        )
    )

    return chart


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    df = load_clean_data()

    chart3 = vis3_productivity_diverging(df)
    chart4 = vis4_commute_wfh_interactive(df)

    base_path = os.path.dirname(os.path.abspath(__file__))

    path3 = os.path.join(base_path, "alex_vis3.html")
    path4 = os.path.join(base_path, "alex_vis4.html")

    chart3.save(path3)
    chart4.save(path4)

    webbrowser.open("file://" + path3)
    webbrowser.open("file://" + path4)

    print("All Alex visualizations generated successfully!")