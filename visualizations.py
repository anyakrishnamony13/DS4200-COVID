import altair as alt
import pandas as pd
import numpy as np
import webbrowser
import os
from cleaning import load_clean_data


# ── SHARED STYLING ────────────────────────────────────────────────────────────
COLORS = {
    "pre":      "#4C72B0",   # muted blue  — Before COVID
    "now":      "#DD8452",   # warm orange — During COVID
    "expected": "#55A868",   # sage green  — Expected After COVID
}
FONT = "Georgia"

INCOME_ORDER = ["Under $35k", "$35k–$74k", "$75k–$124k", "$125k+"]


# ═════════════════════════════════════════════════════════════════════════════
# VIS 1 — Grouped Bar Chart (Altair, Static)
# "The Commute Collapse: WFH Access Before, During, and After COVID by Income"
# ═════════════════════════════════════════════════════════════════════════════

def vis1_wfh_by_income(df):
    """
    Grouped bar chart showing the % of workers with WFH access across three
    time periods (Before / During / Expected After COVID), broken out by
    household income bracket.

    Key takeaway: lower-income workers saw the smallest WFH gains, highlighting
    who was — and wasn't — able to work safely from home.
    """

    # ── 1. SUBSET TO PRE-COVID WORKERS ───────────────────────────────────────
    workers = df[
        df["empl_pre"].str.contains("Employed", na=False) &
        df["income_bracket"].notna()
    ].copy()

    # ── 2. COMPUTE % WITH WFH ACCESS PER INCOME BRACKET × PERIOD ─────────────
    records = []
    for bracket in INCOME_ORDER:
        g = workers[workers["income_bracket"] == bracket]
        n = len(g)

        # Before: wfh_pre == "Yes"
        pct_pre = (g["wfh_pre"] == "Yes").sum() / n * 100

        # During: wfh_now == "Yes"
        pct_now = (g["wfh_now"] == "Yes").sum() / n * 100

        # Expected after: wfh_expect == "Yes" (denominator = those who answered)
        answered = g["wfh_expect"].notna().sum()
        pct_exp = (g["wfh_expect"] == "Yes").sum() / answered * 100 if answered > 0 else np.nan

        records.append({"income_bracket": bracket, "period": "Before COVID",   "pct": pct_pre, "n": n})
        records.append({"income_bracket": bracket, "period": "During COVID",   "pct": pct_now, "n": n})
        records.append({"income_bracket": bracket, "period": "Expected After", "pct": pct_exp, "n": answered})

    plot_df = pd.DataFrame(records)
    plot_df["pct"] = plot_df["pct"].round(1)

    # ── 3. BUILD CHART ────────────────────────────────────────────────────────
    period_order = ["Before COVID", "During COVID", "Expected After"]
    color_range  = [COLORS["pre"], COLORS["now"], COLORS["expected"]]

    # Bars — column facet is NOT set here; it goes on the layer after
    bars = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X(
                "period:N",
                sort=period_order,
                axis=alt.Axis(title=None, labelAngle=0, labelFontSize=11, labelFont=FONT),
            ),
            y=alt.Y(
                "pct:Q",
                scale=alt.Scale(domain=[0, 85]),
                axis=alt.Axis(
                    title="% of Workers with WFH Access",
                    titleFont=FONT,
                    titleFontSize=12,
                    labelFont=FONT,
                    labelFontSize=11,
                    grid=True,
                    gridColor="#e8e8e8",
                    format=".0f",
                ),
            ),
            color=alt.Color(
                "period:N",
                sort=period_order,
                scale=alt.Scale(domain=period_order, range=color_range),
                legend=alt.Legend(
                    title="Time Period",
                    titleFont=FONT,
                    titleFontSize=12,
                    labelFont=FONT,
                    labelFontSize=11,
                    orient="bottom",
                ),
            ),
            tooltip=[
                alt.Tooltip("income_bracket:N", title="Income"),
                alt.Tooltip("period:N",          title="Period"),
                alt.Tooltip("pct:Q",             title="% with WFH Access", format=".1f"),
                alt.Tooltip("n:Q",               title="Workers (n)"),
            ],
        )
        .properties(width=130, height=300)
    )

    # Labels — no column facet here either
    labels = (
        alt.Chart(plot_df)
        .mark_text(
            align="center",
            baseline="bottom",
            dy=-4,
            fontSize=10,
            font=FONT,
            color="#333333",
        )
        .encode(
            x=alt.X("period:N", sort=period_order),
            y=alt.Y("pct:Q"),
            text=alt.Text("pct:Q", format=".0f"),
        )
    )

    # Layer first, THEN facet — Altair requires this order
    chart = (
        alt.layer(bars, labels)
        .facet(
            column=alt.Column(
                "income_bracket:N",
                sort=INCOME_ORDER,
                header=alt.Header(
                    title="Household Income",
                    titleFont=FONT,
                    titleFontSize=13,
                    titleFontWeight="bold",
                    labelFont=FONT,
                    labelFontSize=12,
                    labelPadding=8,
                ),
            ),
        )
        .configure_view(stroke="transparent")
        .configure_axis(domainColor="#cccccc")
        .properties(
            title=alt.TitleParams(
                text="The Commute Collapse: Who Got to Work from Home?",
                subtitle=[
                    "% of pre-COVID workers with WFH access, by household income bracket",
                    "Higher-income workers saw the largest gains — and expect to keep them.",
                ],
                font=FONT,
                fontSize=16,
                fontWeight="bold",
                subtitleFont=FONT,
                subtitleFontSize=12,
                subtitleColor="#555555",
                anchor="start",
                offset=10,
            )
        )
        .resolve_scale(y="shared")
    )

    return chart

# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_clean_data()

    chart1 = vis1_wfh_by_income(df)

    # Save to HTML then open in browser — more reliable than .show() across
    # all Altair versions and environments
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis1.html")
    chart1.save(output_path)
    webbrowser.open("file://" + output_path)
    print(f"Chart saved and opened: {output_path}")