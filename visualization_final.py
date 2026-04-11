"""
visualization_final.py

All 5 project visualizations in one file. Each visualization is its own function and saves to its own HTML output file.

Run with:
    python3 visualization_final.py

Output files:
    vis1.html  — Grouped bar chart: WFH access by income (Altair)
    vis2.html   — Heatmap: transportation mode shift matrix (Altair)
    vis3.html  — Diverging bar: productivity by job category (Altair)
    vis4.html  — Interactive scatter + linked bar: commute vs WFH (Altair)
    vis5.html  — Animated stacked bar: mode migration (D3)
"""

import os
import json
import webbrowser
import altair as alt
import pandas as pd
import numpy as np
from cleaning import load_clean_data


FONT = "Inter"


BG_WHITE     = "#ffffff"
BORDER_COLOR = "#E4EAF0"


TEXT_TITLE    = "#0D1B2A"
TEXT_SUBTITLE = "#5A7A94"
TEXT_AXIS     = "#8FA8BC"
TEXT_SOURCE   = "#B0C4D4"


GRID_COLOR = "#EDF1F5"
AXIS_COLOR = "#D4E0EA"


COLOR_BEFORE   = "#2EC4B6"
COLOR_DURING   = "#4895EF"
COLOR_AFTER    = "#F4A261"
COLOR_INCREASE = "#2EC4B6"
COLOR_DECREASE = "#E63946"

COLOR_EDUC = {
    "High school or less": "#4895EF",
    "Some college":        "#2EC4B6",
    "Bachelor's":          "#F4A261",
    "Graduate degree":     "#9B72CF",
}
EDUC_ORDER   = ["High school or less", "Some college", "Bachelor's", "Graduate degree"]
INCOME_ORDER = ["Under $35k", "$35k\u2013$74k", "$75k\u2013$124k", "$125k+"]
PERIOD_ORDER = ["Before COVID", "During COVID", "Expected After"]

SOURCE_TEXT = "Source: COVID Future Wave 1B Survey, Arizona State University"


_AXIS_CFG = dict(
    labelFont=FONT, labelFontSize=11, labelColor=TEXT_AXIS, labelPadding=6,
    titleFont=FONT, titleFontSize=12, titleColor=TEXT_AXIS, titlePadding=10,
    gridColor=GRID_COLOR, domainColor=AXIS_COLOR, tickColor=AXIS_COLOR,
)

_LEGEND_CFG = dict(
    labelFont=FONT, labelFontSize=12, labelColor=TEXT_AXIS,
    titleFont=FONT, titleFontSize=12, titleColor=TEXT_SUBTITLE,
    titleFontWeight="normal",
    symbolSize=100, symbolStrokeWidth=0,
    padding=8, rowPadding=6,
)

_TITLE_CFG = dict(
    font=FONT, fontSize=18, fontWeight="bold", color=TEXT_TITLE,
    subtitleFont=FONT, subtitleFontSize=12, subtitleColor=TEXT_SUBTITLE,
    subtitleLineHeight=18,
    anchor="start", offset=8,
)
_HEADER_CFG = dict(
    labelFont=FONT, labelFontSize=11, labelColor=TEXT_AXIS,
    titleFont=FONT, titleFontSize=12, titleColor=TEXT_SUBTITLE,
    titleFontWeight="normal",
)


def _add_source(chart_spec):
    """Wrap an Altair chart in a vconcat with a source footnote text mark."""
    source_label = (
        alt.Chart({"values": [{}]})
        .mark_text(
            text=SOURCE_TEXT,
            align="left", baseline="top",
            font=FONT, fontSize=11, color=TEXT_SOURCE,
            dx=2,
        )
        .properties(width=600, height=20)
    )
    return alt.vconcat(chart_spec, source_label, spacing=4)


INTER_FONT_LINK = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">\n'
    '<style>* { font-family: \'Inter\', sans-serif !important; }</style>\n'
)

def _save_and_open(chart, filename):
    """Save an Altair chart to HTML, then open in browser."""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), filename)
    chart.save(path)

    with open(path, "r", encoding="utf-8") as f:
        html = f.read()
    html = html.replace("</head>", INTER_FONT_LINK + "</head>", 1)

    source_html = (
        '<style>'
        'body{margin:16px 24px;}'
        '.vega-embed{display:block;}'
        '#source-line{'
        f'font-family:\'Inter\',sans-serif;font-size:11px;color:{TEXT_SOURCE};'
        f'border-top:1px solid {GRID_COLOR};padding-top:10px;margin-top:6px;'
        'display:table;'
        '}'
        '</style>'
        f'<p id="source-line">{SOURCE_TEXT}</p>'
    )
    html = html.replace("</body>", source_html + "\n</body>", 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    webbrowser.open("file://" + path)
    print(f"Saved and opened: {path}")



def vis1_wfh_by_income(df):
    """
    % of pre-COVID workers with WFH access across Before / During / Expected After,
    grouped by household income bracket.
    """

    workers = df[
        df["empl_pre"].str.contains("Employed", na=False) &
        df["income_bracket"].notna()
    ].copy()

    records = []
    for bracket in INCOME_ORDER:
        g = workers[workers["income_bracket"] == bracket]
        n = len(g)
        pct_pre = (g["wfh_pre"] == "Yes").sum() / n * 100
        pct_now = (g["wfh_now"] == "Yes").sum() / n * 100
        answered = g["wfh_expect"].notna().sum()
        pct_exp  = (g["wfh_expect"] == "Yes").sum() / answered * 100 if answered > 0 else np.nan
        records += [
            {"income_bracket": bracket, "period": "Before COVID",   "pct": round(pct_pre, 1), "n": n},
            {"income_bracket": bracket, "period": "During COVID",   "pct": round(pct_now, 1), "n": n},
            {"income_bracket": bracket, "period": "Expected After", "pct": round(pct_exp, 1), "n": answered},
        ]
    plot_df = pd.DataFrame(records)

    bars = (
        alt.Chart(plot_df)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("period:N", sort=PERIOD_ORDER,
                    axis=alt.Axis(title=None, labels=False, ticks=False)),
            y=alt.Y("pct:Q", scale=alt.Scale(domain=[0, 85]),
                    axis=alt.Axis(title="% of workers with WFH access",
                                  format=".0f", grid=True,
                                  labelPadding=6, titlePadding=10)),
            color=alt.Color(
                "period:N", sort=PERIOD_ORDER,
                scale=alt.Scale(domain=PERIOD_ORDER,
                                range=[COLOR_BEFORE, COLOR_DURING, COLOR_AFTER]),
                legend=alt.Legend(title="Time period", orient="right",
                                  **{k: v for k, v in _LEGEND_CFG.items()
                                     if k not in ("padding", "rowPadding")}),
            ),
            tooltip=[
                alt.Tooltip("income_bracket:N", title="Income"),
                alt.Tooltip("period:N",         title="Period"),
                alt.Tooltip("pct:Q",            title="% with WFH access", format=".1f"),
                alt.Tooltip("n:Q",              title="Workers (n)"),
            ],
        )
        .properties(width=120, height=260)
    )

    chart = (
        alt.layer(bars)
        .facet(
            column=alt.Column(
                "income_bracket:N", sort=INCOME_ORDER,
                header=alt.Header(
                    title="Household income",
                    labelPadding=8,
                    **{**_HEADER_CFG, "titleFontWeight": "bold"},
                ),
            )
        )
        .configure_view(stroke=BORDER_COLOR, strokeWidth=1)
        .configure_axis(**_AXIS_CFG)
        .configure_legend(**_LEGEND_CFG)
        .configure_header(**_HEADER_CFG)
        .configure_title(**_TITLE_CFG)
        .properties(
            title=alt.TitleParams(
                text="The commute collapse: who got to work from home?",
                subtitle=[
                    "% of pre-COVID workers with WFH access, by household income."
                ],
                subtitleColor=TEXT_SUBTITLE,
            )
        )
        .resolve_scale(y="shared")
    )
    return chart


def vis2_mode_shift_heatmap(df, group_col="income_bracket"):
    """
    Altair heatmap of average change in transportation use (During − Before COVID).
    """

    PRE_TO_DAYS = {0: 0, 1: 0.1, 2: 0.5, 3: 2.5, 4: 7}

    MODE_MAP = {
        "privateveh": "Private Vehicle", "ridehail":   "Ridehail/Taxi",
        "transit":    "Transit",          "bikepers":   "Personal Bike",
        "bikeshared": "Shared Bike",      "walk":       "Walking",
    }
    MODE_ORDER = ["Private Vehicle", "Ridehail/Taxi", "Transit",
                  "Personal Bike", "Shared Bike", "Walking"]

    if group_col == "age_group":
        group_order = ["18\u201329", "30\u201344", "45\u201359", "60\u201374", "75+"]
        group_title = "Age group"
    else:
        group_order = INCOME_ORDER
        group_title = "Household income"

    temp_df = df[df[group_col].notna()].copy()
    records = []

    for mode_code, mode_label in MODE_MAP.items():
        pre_col = f"tr_freq_pre_{mode_code}"
        now_col = f"tr_freq_now_{mode_code}"
        temp = temp_df[[group_col, pre_col, now_col, "weight"]].copy()
        temp = temp.dropna(subset=[pre_col, now_col, "weight"])
        if temp.empty:
            continue

        temp[pre_col] = temp[pre_col].map(PRE_TO_DAYS)
        temp = temp.dropna(subset=[pre_col])
        temp["change"] = temp[now_col] - temp[pre_col]
        for group, g in temp.groupby(group_col, observed=False):
            if len(g) == 0:
                continue
            avg_change = np.average(g["change"], weights=g["weight"])
            records.append({
                "group": str(group),
                "mode":  mode_label,
                "avg_change": round(avg_change, 2),
            })

    plot_df = pd.DataFrame(records)
    max_abs = plot_df["avg_change"].abs().max()


    heatmap = (
        alt.Chart(plot_df)
        .mark_rect(stroke=BG_WHITE, strokeWidth=3)
        .encode(
            x=alt.X("group:N", sort=group_order,
                    axis=alt.Axis(title=group_title, labelAngle=0,
                                  labelPadding=8, titlePadding=12)),
            y=alt.Y("mode:N", sort=MODE_ORDER,
                    axis=alt.Axis(title="Transportation mode",
                                  labelPadding=8, titlePadding=12)),
            color=alt.Color(
                "avg_change:Q",
                scale=alt.Scale(
                    domain=[-max_abs, -max_abs/2, 0, max_abs/2, max_abs],
                    range=[
                        "#C0392B",
                        "#F5A8AC",
                        "#F7F9FC",
                        "#7ED4CD",
                        COLOR_BEFORE,
                    ],
                    interpolate="lab",
                ),
                legend=alt.Legend(
                    title="Avg change in days/week",
                    gradientLength=200,
                    gradientThickness=16,
                    labelPadding=6,
                    titlePadding=8,
                    values=[-1.0, -0.5, 0.0, 0.5, 1.0],
                    labelExpr="format(datum.value, '+.1f')",
                ),
            ),
            tooltip=[
                alt.Tooltip("group:N",      title=group_title),
                alt.Tooltip("mode:N",       title="Mode"),
                alt.Tooltip("avg_change:Q", title="Avg change (days/wk)", format="+.2f"),
            ],
        )
        .properties(width=500, height=280)
    )

    cell_labels = (
        alt.Chart(plot_df)
        .mark_text(font=FONT, fontSize=12, fontWeight="bold")
        .encode(
            x=alt.X("group:N", sort=group_order),
            y=alt.Y("mode:N",  sort=MODE_ORDER),
            text=alt.Text("avg_change:Q", format="+.2f"),
            color=alt.condition(
                "abs(datum.avg_change) > 0.25",
                alt.value("#ffffff"),
                alt.value(TEXT_TITLE),
            ),
        )
    )

    chart = (
        alt.layer(heatmap, cell_labels)
        .configure(padding={"top": 20, "right": 160, "bottom": 20, "left": 20})
        .configure_view(stroke=BORDER_COLOR, strokeWidth=1)
        .configure_axis(**_AXIS_CFG)
        .configure_legend(**_LEGEND_CFG)
        .configure_title(**_TITLE_CFG)
        .properties(
            title=alt.TitleParams(
                text="Mode shift matrix: which transportation modes gained and lost users?",
                subtitle=[
                    "Avg change in days/week of use (During COVID \u2212 Before COVID), by income group."
                ],
                subtitleColor=TEXT_SUBTITLE,
            )
        )
    )
    return chart


def vis3_productivity_diverging(df):
    """
    % reporting increased vs decreased productivity during COVID, by job category.
    Color = binary categorical (teal = increased, red = decreased).
    Legend on the right, matching vis5 layout.
    """

    def _bucket(x):
        x = str(x).lower()
        if "increase" in x: return "Increased"
        if "decrease" in x: return "Decreased"
        return None

    data = df[df["jobcat_pre"].notna()].copy()
    data["direction"] = data["prod_change"].apply(_bucket)
    data = data[data["direction"].notna()]

    grouped = (
        data.groupby(["jobcat_pre", "direction"])
        .size().reset_index(name="count")
    )
    totals = grouped.groupby("jobcat_pre")["count"].transform("sum")
    grouped["pct"] = (grouped["count"] / totals * 100).round(1)
    grouped["pct_diverge"] = grouped.apply(
        lambda r: -r["pct"] if r["direction"] == "Decreased" else r["pct"], axis=1
    )

    net_order = (
        grouped.groupby("jobcat_pre")
        .apply(lambda g: g.loc[g["direction"] == "Increased", "pct"].sum()
                       - g.loc[g["direction"] == "Decreased", "pct"].sum())
        .sort_values(ascending=False).index.tolist()
    )

    chart = (
        alt.Chart(grouped)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3,
                  cornerRadiusTopLeft=3, cornerRadiusBottomLeft=3)
        .encode(
            y=alt.Y("jobcat_pre:N", sort=net_order,
                    axis=alt.Axis(title="Job category",
                                  labelPadding=8, titlePadding=12)),
            x=alt.X("pct_diverge:Q",
                    axis=alt.Axis(title="% reporting productivity change",
                                  format=".0f", grid=True,
                                  labelPadding=6, titlePadding=10)),
            color=alt.Color(
                "direction:N",
                scale=alt.Scale(domain=["Increased", "Decreased"],
                                range=[COLOR_INCREASE, COLOR_DECREASE]),
                legend=alt.Legend(title="Productivity change", orient="right",
                                  **{k: v for k, v in _LEGEND_CFG.items()
                                     if k not in ("padding", "rowPadding")}),
            ),
            tooltip=[
                alt.Tooltip("jobcat_pre:N",  title="Job category"),
                alt.Tooltip("direction:N",   title="Direction"),
                alt.Tooltip("pct:Q",         title="% of job category", format=".1f"),
                alt.Tooltip("count:Q",       title="Respondents (n)"),
            ],
        )
        .properties(width=580, height=260)
        .configure(padding={"top": 20, "right": 160, "bottom": 20, "left": 20})
        .configure_view(stroke=BORDER_COLOR, strokeWidth=1)
        .configure_axis(**_AXIS_CFG)
        .configure_legend(**_LEGEND_CFG)
        .configure_title(**_TITLE_CFG)
        .properties(
            title=alt.TitleParams(
                text="Productivity winners and losers: who thrived working from home?",
                subtitle=[
                    "Share of workers reporting increased vs. decreased productivity during COVID, by job category."
                ],
                subtitleColor=TEXT_SUBTITLE,
            )
        )
    )
    return chart



def vis4_commute_wfh_interactive(df):
    """
    Scatter (commute time vs WFH days) + linked bar (expected WFH freq).
    Brush selection filters the bar chart.
    Color = education level (4 distinct hues).
    Legend on right, matching vis5.
    """

    data = df[
        df["pre_work_pri_time"].notna() &
        df["wfh_now_days"].notna() &
        df["wfh_freq_exp"].notna() &
        df["educ_simple"].notna()
    ].copy()
    data = data[data["pre_work_pri_time"] <= 120].copy()

    brush = alt.selection_interval(name="brush")

    WFH_EXP_ORDER = ["Never", "A few times/year", "A few times/month",
                     "Once/week", "A few times/week", "Every day"]

    WFH_EXP_LABELS = {
        "Never": "Never",
        "A few times/year": "Few times a year",
        "A few times/month": "Few times a month",
        "Once/week": "Once a week",
        "A few times/week": "Few times a week",
        "Every day": "Every day",
    }
    data["wfh_freq_exp_label"] = data["wfh_freq_exp"].map(WFH_EXP_LABELS).fillna(data["wfh_freq_exp"])

    educ_scale = alt.Scale(
        domain=EDUC_ORDER,
        range=[COLOR_EDUC[e] for e in EDUC_ORDER],
    )

    legend_cfg = {k: v for k, v in _LEGEND_CFG.items()
                  if k not in ("padding", "rowPadding")}

    scatter = (
        alt.Chart(data)
        .mark_circle(size=45, opacity=0.55, stroke="white", strokeWidth=0.5)
        .encode(
            x=alt.X("pre_work_pri_time:Q",
                    title="Pre-COVID commute time (minutes)",
                    scale=alt.Scale(domain=[0, 120]),
                    axis=alt.Axis(labelPadding=6, titlePadding=10)),
            y=alt.Y("wfh_now_days:Q",
                    title="WFH days in past 7 days",
                    axis=alt.Axis(labelPadding=6, titlePadding=10)),
            color=alt.Color("educ_simple:N", sort=EDUC_ORDER, scale=educ_scale,
                            legend=alt.Legend(title="Education level", orient="right",
                                              **legend_cfg)),
            tooltip=[
                alt.Tooltip("pre_work_pri_time:Q", title="Commute time (min)"),
                alt.Tooltip("wfh_now_days:Q",      title="WFH days now"),
                alt.Tooltip("wfh_freq_exp:N",      title="Expected WFH freq"),
                alt.Tooltip("educ_simple:N",        title="Education"),
            ],
        )
        .add_params(brush)
        .properties(width=620, height=280)
    )

    bars = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3)
        .encode(
            x=alt.X("wfh_freq_exp_label:N",
                    sort=[WFH_EXP_LABELS[v] for v in WFH_EXP_ORDER],
                    axis=alt.Axis(
                        title="Expected WFH frequency after COVID",
                        labelAngle=0,
                        labelLimit=200,
                        labelPadding=8,
                        titlePadding=14,
                    )),
            y=alt.Y("count()",
                    title="Number of respondents",
                    axis=alt.Axis(labelPadding=6, titlePadding=10)),
            color=alt.Color("educ_simple:N", sort=EDUC_ORDER,
                            scale=educ_scale, legend=None),
            tooltip=[
                alt.Tooltip("wfh_freq_exp:N", title="Expected freq"),
                alt.Tooltip("count()",        title="Respondents"),
            ],
        )
        .transform_filter(brush)
        .properties(width=620, height=160)
    )

    chart = (
        alt.vconcat(scatter, bars, spacing=24)
        .configure(padding={"top": 20, "right": 160, "bottom": 20, "left": 20})
        .configure_view(stroke=BORDER_COLOR, strokeWidth=1)
        .configure_axis(**_AXIS_CFG)
        .configure_legend(**_LEGEND_CFG)
        .configure_title(**_TITLE_CFG)
        .properties(
            title=alt.TitleParams(
                text="Commute time vs. WFH adoption: did long commuters embrace remote work more?",
                subtitle=[
                    "Each dot represents a worker, drag to select a region and the bar chart below will update",
                    "to show expected post-COVID WFH frequency for the selected group."
                ],
                subtitleColor=TEXT_SUBTITLE,
            )
        )
    )
    return chart


def _build_vis5_data(df):
    """Compute weighted percentages for the D3 stacked bar chart."""
    MODE_MAP = {
        "privateveh": "Private Vehicle", "ridehail": "Ridehail/Taxi",
        "transit": "Transit",            "bikepers": "Personal Bike",
        "bikeshared": "Shared Bike",     "walk": "Walking",
    }
    TIER_ORDER     = ["Never", "Rarely", "Sometimes", "Often"]
    EXP_TIER_ORDER = ["Much less", "Somewhat less", "About the same",
                      "Somewhat more", "Much more"]

    def _usage(x):
        if pd.isna(x): return np.nan
        if x == 0:    return "Never"
        if x <= 1:    return "Rarely"
        if x <= 3:    return "Sometimes"
        return "Often"

    def _exp(x):
        if pd.isna(x): return np.nan
        return {-2: "Much less", -1: "Somewhat less", 0: "About the same",
                 1: "Somewhat more", 2: "Much more"}.get(x, np.nan)

    records = []
    for mode_code, mode_label in MODE_MAP.items():
        pre_col = f"tr_freq_pre_{mode_code}"
        now_col = f"tr_freq_now_{mode_code}"
        exp_col = f"tr_freq_exp_{mode_code}"

        pre = df[[pre_col, "weight"]].copy()
        pre["tier"] = pre[pre_col].apply(_usage)
        pre = pre.dropna(subset=["tier", "weight"])
        total = pre["weight"].sum()
        for tier in TIER_ORDER:
            w = pre.loc[pre["tier"] == tier, "weight"].sum()
            records.append({"period": "Before COVID", "mode": mode_label,
                            "tier": tier, "percent": round(w / total * 100, 2)})

        now = df[[now_col, "weight"]].copy()
        now["tier"] = now[now_col].apply(_usage)
        now = now.dropna(subset=["tier", "weight"])
        total = now["weight"].sum()
        for tier in TIER_ORDER:
            w = now.loc[now["tier"] == tier, "weight"].sum()
            records.append({"period": "During COVID", "mode": mode_label,
                            "tier": tier, "percent": round(w / total * 100, 2)})

        exp = df[[exp_col, "weight"]].copy()
        exp["tier"] = exp[exp_col].apply(_exp)
        exp = exp.dropna(subset=["tier", "weight"])
        total = exp["weight"].sum()
        for tier in EXP_TIER_ORDER:
            w = exp.loc[exp["tier"] == tier, "weight"].sum()
            records.append({"period": "Expected After", "mode": mode_label,
                            "tier": tier, "percent": round(w / total * 100, 2)})

    return records


def vis5_mode_migration_d3(df, output_filename="vis5.html"):
    """
    D3 interactive stacked bar with animated transitions.
    """
    data = _build_vis5_data(df)
    data_json = json.dumps(data)

    usage_colors_js    = json.dumps(["#EAF3FB", "#8FC8F0", "#4895EF", "#1a5fa8"])
    expected_colors_js = json.dumps(["#E63946", "#F5A8AC", "#EDF1F5", "#7ED4CD", "#2EC4B6"])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>The great mode migration</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; }}
    body {{
      font-family: 'Inter', sans-serif;
      margin: 32px 40px;
      background: {BG_WHITE};
      color: {TEXT_TITLE};
    }}
    .chart-title {{
      font-size: 18px;
      font-weight: 600;
      color: {TEXT_TITLE};
      margin: 0 0 5px;
    }}
    .chart-subtitle {{
      font-size: 12px;
      color: {TEXT_SUBTITLE};
      margin: 0 0 20px;
      max-width: 800px;
      line-height: 1.5;
    }}
    .buttons {{
      display: flex;
      gap: 10px;
      margin-bottom: 20px;
    }}
    button {{
      font-family: 'Inter', sans-serif;
      font-size: 13px;
      font-weight: 500;
      padding: 8px 18px;
      border: 1.5px solid {BORDER_COLOR};
      border-radius: 6px;
      background: {BG_WHITE};
      color: {TEXT_SUBTITLE};
      cursor: pointer;
      transition: all 0.15s;
    }}
    button:hover {{
      border-color: {COLOR_DURING};
      color: {COLOR_DURING};
    }}
    button.active {{
      background: {COLOR_DURING};
      border-color: {COLOR_DURING};
      color: #ffffff;
    }}
    .scale-warning {{
      display: none;
      background: #FFF8EC;
      border: 1px solid {COLOR_AFTER};
      border-radius: 6px;
      padding: 10px 16px;
      margin-bottom: 16px;
      max-width: 800px;
      font-size: 12px;
      color: {TEXT_SUBTITLE};
      line-height: 1.5;
    }}
    .tooltip {{
      position: fixed;
      opacity: 0;
      background: {BG_WHITE};
      border: 1px solid {BORDER_COLOR};
      border-radius: 6px;
      padding: 10px 14px;
      pointer-events: none;
      font-size: 12px;
      line-height: 1.6;
      color: {TEXT_TITLE};
      box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }}
    .source {{
      font-size: 11px;
      color: {TEXT_SOURCE};
      margin-top: 16px;
      border-top: 1px solid {GRID_COLOR};
      padding-top: 10px;
      max-width: 840px;
    }}
  </style>
</head>
<body>
  <h2 class="chart-title">The great mode migration: how transportation choices shifted</h2>
  <p class="chart-subtitle">
    Toggle between time periods to see how the share of respondents using each
    transportation mode at each frequency tier changed before, during, and after COVID.
  </p>

  <div class="scale-warning" id="scaleWarning">
    <strong>Note:</strong> The "Expected After" view uses a different scale.
    Bars show <em>expected change relative to before COVID</em> (Much less to Much more),
    not absolute usage frequency. Avoid direct visual comparison with the other periods.
  </div>

  <div class="buttons">
    <button id="btn-before" class="active" onclick="updateChart('Before COVID', this)">Before COVID</button>
    <button id="btn-during" onclick="updateChart('During COVID', this)">During COVID</button>
    <button id="btn-after"  onclick="updateChart('Expected After', this)">Expected After</button>
  </div>

  <svg id="chart"></svg>
  <div class="tooltip" id="tooltip"></div>
  <p class="source">{SOURCE_TEXT}</p>

  <script>
    const rawData = {data_json};

    const margin = {{ top: 16, right: 200, bottom: 80, left: 68 }};
    const totalW = 840, totalH = 420;
    const W = totalW - margin.left - margin.right;
    const H = totalH - margin.top  - margin.bottom;

    const svg = d3.select("#chart")
      .attr("width",  totalW)
      .attr("height", totalH);

    const g = svg.append("g")
      .attr("transform", `translate(${{margin.left}},${{margin.top}})`);

    const modes = ["Private Vehicle","Ridehail/Taxi","Transit",
                   "Personal Bike","Shared Bike","Walking"];
    const usageTiers    = ["Never","Rarely","Sometimes","Often"];
    const expectedTiers = ["Much less","Somewhat less","About the same",
                           "Somewhat more","Much more"];

    const usageColor = d3.scaleOrdinal()
      .domain(usageTiers)
      .range({usage_colors_js});

    const expectedColor = d3.scaleOrdinal()
      .domain(expectedTiers)
      .range({expected_colors_js});

    const x = d3.scaleBand().domain(modes).range([0, W]).padding(0.25);
    const y = d3.scaleLinear().domain([0, 100]).range([H, 0]);

    g.selectAll(".gridline")
      .data([0, 25, 50, 75, 100]).join("line")
      .attr("class", "gridline")
      .attr("x1", 0).attr("x2", W)
      .attr("y1", d => y(d)).attr("y2", d => y(d))
      .attr("stroke", "{GRID_COLOR}").attr("stroke-width", 1);

    g.append("g")
      .attr("transform", `translate(0,${{H}})`)
      .call(d3.axisBottom(x).tickSize(0))
      .call(ax => ax.select(".domain").attr("stroke", "{AXIS_COLOR}"))
      .selectAll("text")
      .attr("dy", "1.2em")
      .style("font-family","Inter").style("font-size","12px")
      .style("fill", "{TEXT_AXIS}");

    g.append("g")
      .call(d3.axisLeft(y).ticks(5).tickFormat(d => d + "%").tickSize(0))
      .call(ax => ax.select(".domain").attr("stroke", "{AXIS_COLOR}"))
      .selectAll("text")
      .style("font-family","Inter").style("font-size","11px")
      .style("fill", "{TEXT_AXIS}");

    svg.append("text")
      .attr("transform","rotate(-90)")
      .attr("x", -(margin.top + H / 2))
      .attr("y", 18)
      .attr("text-anchor","middle")
      .style("font-family","Inter").style("font-size","12px")
      .style("fill", "{TEXT_AXIS}")
      .text("% of respondents");

    svg.append("text")
      .attr("x", margin.left + W / 2)
      .attr("y", totalH - 8)
      .attr("text-anchor","middle")
      .style("font-family","Inter").style("font-size","12px")
      .style("fill", "{TEXT_AXIS}")
      .text("Transportation mode");

    const legend = svg.append("g").attr("id","legend")
      .attr("transform", `translate(${{margin.left + W + 20}},${{margin.top + 8}})`);

    const tooltip = d3.select("#tooltip");

    function drawLegend(period) {{
      legend.selectAll("*").remove();
      const tiers  = period === "Expected After" ? expectedTiers : usageTiers;
      const cScale = period === "Expected After" ? expectedColor : usageColor;
      const label  = period === "Expected After" ? "Expected change" : "Usage frequency";

      legend.append("text").attr("x",0).attr("y",0)
        .style("font-size","12px").style("font-weight","500")
        .style("fill","{TEXT_SUBTITLE}").style("font-family","Inter").text(label);

      tiers.forEach((tier, i) => {{
        const row = legend.append("g").attr("transform",`translate(0,${{i*26+18}})`);
        row.append("rect").attr("width",13).attr("height",13).attr("rx",2)
          .attr("fill", cScale(tier))
          .attr("stroke","{BORDER_COLOR}").attr("stroke-width",0.5);
        row.append("text").attr("x",20).attr("y",11)
          .style("font-size","12px").style("fill","{TEXT_AXIS}")
          .style("font-family","Inter").text(tier);
      }});
    }}

    function prepareData(period) {{
      const tiers    = period === "Expected After" ? expectedTiers : usageTiers;
      const filtered = rawData.filter(d => d.period === period);
      const modeMap  = new Map();
      modes.forEach(m => {{
        modeMap.set(m, {{ mode: m }});
        tiers.forEach(t => modeMap.get(m)[t] = 0);
      }});
      filtered.forEach(d => {{ if (modeMap.has(d.mode)) modeMap.get(d.mode)[d.tier] = +d.percent; }});
      return {{ data: Array.from(modeMap.values()), tiers }};
    }}

    function updateChart(period, btn) {{
      document.querySelectorAll(".buttons button").forEach(b => b.classList.remove("active"));
      if (btn) btn.classList.add("active");
      document.getElementById("scaleWarning").style.display =
        period === "Expected After" ? "block" : "none";

      const {{ data, tiers }} = prepareData(period);
      const cScale  = period === "Expected After" ? expectedColor : usageColor;
      const stacked = d3.stack().keys(tiers)(data);

      drawLegend(period);

      const layers = g.selectAll(".layer").data(stacked, d => d.key);
      layers.exit().remove();

      const layerMerge = layers.enter().append("g").attr("class","layer")
        .merge(layers).attr("fill", d => cScale(d.key));

      const rects = layerMerge.selectAll("rect")
        .data(d => d.map(v => ({{ ...v, key: d.key }})), d => d.data.mode);

      rects.exit().transition().duration(400)
        .attr("height", 0).attr("y", H).remove();

      rects.enter().append("rect")
        .attr("x", d => x(d.data.mode)).attr("width", x.bandwidth())
        .attr("y", H).attr("height", 0).attr("rx", 2)
        .on("mousemove", (event, d) => {{
          tooltip.style("opacity",1)
            .html(`<strong>${{d.data.mode}}</strong><br>${{d.key}}: ${{(d[1]-d[0]).toFixed(1)}}%`)
            .style("left",(event.clientX+14)+"px")
            .style("top", (event.clientY-36)+"px");
        }})
        .on("mouseout", () => tooltip.style("opacity",0))
        .merge(rects)
        .transition().duration(700).ease(d3.easeCubicInOut)
        .attr("x", d => x(d.data.mode)).attr("width", x.bandwidth())
        .attr("y", d => y(d[1])).attr("height", d => Math.max(0, y(d[0])-y(d[1])));
    }}

    updateChart("Before COVID", document.getElementById("btn-before"));
  </script>
</body>
</html>"""

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"Saved: {out_path}")
    return out_path

if __name__ == "__main__":
    df = load_clean_data()

    _save_and_open(vis1_wfh_by_income(df), "vis1.html")
    _save_and_open(vis2_mode_shift_heatmap(df), "vis2.html")
    _save_and_open(vis3_productivity_diverging(df), "vis3.html")
    _save_and_open(vis4_commute_wfh_interactive(df), "vis4.html")
    vis5_path = vis5_mode_migration_d3(df, "vis5.html")
    webbrowser.open("file://" + vis5_path)
