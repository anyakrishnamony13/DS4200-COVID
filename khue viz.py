import pandas as pd
import numpy as np
import os
import json
import webbrowser
from cleaning import load_clean_data


# ── SHARED SETTINGS ──────────────────────────────────────────────────────────
FONT = "Georgia"

MODE_MAP = {
    "privateveh": "Private Vehicle",
    "ridehail": "Ridehail/Taxi",
    "transit": "Transit",
    "bikepers": "Personal Bike",
    "bikeshared": "Shared Bike",
    "walk": "Walking",
}

MODE_ORDER = [
    "Private Vehicle",
    "Ridehail/Taxi",
    "Transit",
    "Personal Bike",
    "Shared Bike",
    "Walking",
]

AGE_ORDER = ["18–29", "30–44", "45–59", "60–74", "75+"]
INCOME_ORDER = ["Under $35k", "$35k–$74k", "$75k–$124k", "$125k+"]
TIER_ORDER = ["Never", "Rarely", "Sometimes", "Often"]
EXP_TIER_ORDER = [
    "Much less",
    "Somewhat less",
    "About the same",
    "Somewhat more",
    "Much more",
]

# Ordinal-to-days mapping for pre-COVID frequency columns
# Original encoding: 0=Never, 1=A few times/year, 2=A few times/month,
#                    3=A few times/week, 4=Every day
PRE_TO_DAYS = {0: 0, 1: 0.1, 2: 0.5, 3: 2.5, 4: 7}


# ═════════════════════════════════════════════════════════════════════════════
# STATIC 2 — HEATMAP
# ═════════════════════════════════════════════════════════════════════════════

def vis2_mode_shift_heatmap(df, group_col="income_bracket"):
    """
    Matplotlib heatmap of average change in transportation use
    from Before COVID to During COVID.

    Pre-COVID columns use an ordinal frequency scale (0–4) which is
    converted to approximate days-per-week before computing the change,
    so that both axes are on a comparable 0–7 scale.
    """
    import matplotlib.pyplot as plt

    if group_col == "age_group":
        group_order = AGE_ORDER
        group_title = "Age Group"
    elif group_col == "income_bracket":
        group_order = INCOME_ORDER
        group_title = "Household Income"
    else:
        group_order = sorted(df[group_col].dropna().astype(str).unique())
        group_title = group_col

    temp_df = df[df[group_col].notna()].copy()
    records = []

    for mode_code, mode_label in MODE_MAP.items():
        pre_col = f"tr_freq_pre_{mode_code}"
        now_col = f"tr_freq_now_{mode_code}"

        temp = temp_df[[group_col, pre_col, now_col, "weight"]].copy()
        temp = temp.dropna(subset=[pre_col, now_col, "weight"])

        if temp.empty:
            continue

        # ── FIX: normalize pre ordinal (0–4) to days-per-week (0–7) ──────
        # Check your encoding first with: df[pre_col].value_counts().sort_index()
        # If values are NOT 0–4, update PRE_TO_DAYS at the top of this file.
        temp[pre_col] = temp[pre_col].map(PRE_TO_DAYS)
        temp = temp.dropna(subset=[pre_col])  # drop any unmapped values
        temp["change"] = temp[now_col] - temp[pre_col]
        # ─────────────────────────────────────────────────────────────────

        for group, g in temp.groupby(group_col, observed=False):
            if len(g) == 0:
                continue

            avg_change = np.average(g["change"], weights=g["weight"])
            records.append({
                "group": str(group),
                "mode": mode_label,
                "avg_change": avg_change
            })

    plot_df = pd.DataFrame(records)

    pivot = (
        plot_df.pivot(index="mode", columns="group", values="avg_change")
        .reindex(index=MODE_ORDER, columns=group_order)
    )

    fig, ax = plt.subplots(figsize=(9, 4.8))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto")

    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, fontsize=10, fontname=FONT)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10, fontname=FONT)

    ax.set_xlabel(group_title, fontsize=12, fontname=FONT)
    ax.set_ylabel("Transportation Mode", fontsize=12, fontname=FONT)
    ax.set_title(
        "Mode Shift Matrix: Which Transportation Modes Gained and Lost Users?",
        fontsize=15,
        fontname=FONT,
        weight="bold",
        pad=14
    )

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center",
                    color="black", fontsize=10,
                    fontname=FONT, fontweight="bold"
                )


    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Avg Change in Days/Week (During − Before)", fontsize=11, fontname=FONT)

    plt.tight_layout()
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# INTERACTIVE 5 — D3 STACKED / ANIMATED BAR
# BEFORE / DURING / EXPECTED AFTER
# ═════════════════════════════════════════════════════════════════════════════

def categorize_usage(x):
    """Convert days-per-week (now columns) or ordinal (pre columns, already
    mapped) into usage tier labels."""
    if pd.isna(x):
        return np.nan
    if x == 0:
        return "Never"
    elif x <= 1:
        return "Rarely"
    elif x <= 3:
        return "Sometimes"
    else:
        return "Often"


def categorize_expected(x):
    if pd.isna(x):
        return np.nan
    exp_map = {
        -2: "Much less",
        -1: "Somewhat less",
         0: "About the same",
         1: "Somewhat more",
         2: "Much more",
    }
    return exp_map.get(x, np.nan)


def build_vis5_data(df):
    """
    Build weighted percentages for the stacked animated bar chart.
    Before/During = usage tiers (Never / Rarely / Sometimes / Often)
    Expected After = expected change tiers (Much less … Much more)
    """
    records = []

    for mode_code, mode_label in MODE_MAP.items():
        pre_col = f"tr_freq_pre_{mode_code}"
        now_col = f"tr_freq_now_{mode_code}"
        exp_col = f"tr_freq_exp_{mode_code}"

        # ── Before ────────────────────────────────────────────────────────
        pre = df[[pre_col, "weight"]].copy()
        # Map ordinal to days so categorize_usage thresholds are consistent
        pre[pre_col] = pre[pre_col].map(PRE_TO_DAYS)
        pre["tier"] = pre[pre_col].apply(categorize_usage)
        pre = pre.dropna(subset=["tier", "weight"])

        total_pre = pre["weight"].sum()
        for tier in TIER_ORDER:
            tier_weight = pre.loc[pre["tier"] == tier, "weight"].sum()
            pct = (tier_weight / total_pre * 100) if total_pre > 0 else 0
            records.append({
                "period": "Before COVID",
                "mode": mode_label,
                "tier": tier,
                "percent": round(pct, 2)
            })

        # ── During ────────────────────────────────────────────────────────
        now = df[[now_col, "weight"]].copy()
        now["tier"] = now[now_col].apply(categorize_usage)
        now = now.dropna(subset=["tier", "weight"])

        total_now = now["weight"].sum()
        for tier in TIER_ORDER:
            tier_weight = now.loc[now["tier"] == tier, "weight"].sum()
            pct = (tier_weight / total_now * 100) if total_now > 0 else 0
            records.append({
                "period": "During COVID",
                "mode": mode_label,
                "tier": tier,
                "percent": round(pct, 2)
            })

        # ── Expected After ────────────────────────────────────────────────
        exp = df[[exp_col, "weight"]].copy()
        exp["tier"] = exp[exp_col].apply(categorize_expected)
        exp = exp.dropna(subset=["tier", "weight"])

        total_exp = exp["weight"].sum()
        for tier in EXP_TIER_ORDER:
            tier_weight = exp.loc[exp["tier"] == tier, "weight"].sum()
            pct = (tier_weight / total_exp * 100) if total_exp > 0 else 0
            records.append({
                "period": "Expected After",
                "mode": mode_label,
                "tier": tier,
                "percent": round(pct, 2)
            })

    return records


def export_vis5_html(data, output_filename="vis5.html"):
    """
    Export D3 HTML with data embedded directly in the page.
    """
    data_json = json.dumps(data)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>How Transportation Mode Use Shifted Before, During, and After COVID</title>
  <script src="https://d3js.org/d3.v7.min.js"></script>
  <style>
    body {{
      font-family: Georgia, serif;
      margin: 32px;
      color: #222;
    }}

    h2 {{
      margin-bottom: 4px;
    }}

    .subtitle {{
      margin-top: 0;
      margin-bottom: 6px;
      color: #555;
      max-width: 980px;
      line-height: 1.4;
    }}

    /* ── scale-change warning banner ── */
    .scale-warning {{
      display: none;
      background: #fff3cd;
      border: 1px solid #ffc107;
      padding: 8px 14px;
      margin-bottom: 14px;
      max-width: 980px;
      font-size: 13px;
      line-height: 1.4;
    }}

    .buttons {{
      margin-bottom: 18px;
    }}

    button {{
      font-family: Georgia, serif;
      font-size: 14px;
      padding: 8px 14px;
      margin-right: 10px;
      border: 1px solid #bbb;
      background: white;
      cursor: pointer;
    }}

    button.active {{
      background: #e8e8e8;
      font-weight: bold;
      border-color: #888;
    }}

    button:hover {{
      background: #f5f5f5;
    }}

    .tooltip {{
      position: absolute;
      opacity: 0;
      background: white;
      border: 1px solid #ccc;
      padding: 8px 10px;
      pointer-events: none;
      font-size: 13px;
      line-height: 1.35;
    }}
  </style>
</head>
<body>
  <h2>How Transportation Mode Use Shifted Before, During, and After COVID</h2>
  <p class="subtitle">
    <strong>Before</strong> and <strong>During</strong> show what share of respondents
    used each mode at each frequency tier. <strong>Expected After</strong> shows whether
    respondents expect their use to increase or decrease relative to before COVID.
  </p>

  <!-- Warning shown only for Expected After -->
  <div class="scale-warning" id="scaleWarning">
    ⚠️ <strong>Note:</strong> The "Expected After" view uses a different scale than
    Before/During. Bars here show <em>expected change relative to before COVID</em>
    (e.g. "Much less" to "Much more"), not absolute usage frequency.
    Direct visual comparison with the other two periods should be made cautiously.
  </div>

  <div class="buttons">
    <button id="btn-before" class="active" onclick="updateChart('Before COVID', this)">Before COVID</button>
    <button id="btn-during" onclick="updateChart('During COVID', this)">During COVID</button>
    <button id="btn-after" onclick="updateChart('Expected After', this)">Expected After</button>
  </div>

  <svg width="1040" height="560"></svg>
  <div class="tooltip"></div>

  <script>
    const rawData = {data_json};

    const svg = d3.select("svg");
    const tooltip = d3.select(".tooltip");

    const margin = {{ top: 30, right: 220, bottom: 95, left: 80 }};
    const width = +svg.attr("width") - margin.left - margin.right;
    const height = +svg.attr("height") - margin.top - margin.bottom;

    const g = svg.append("g")
      .attr("transform", `translate(${{margin.left}}, ${{margin.top}})`);

    const usageTiers    = ["Never", "Rarely", "Sometimes", "Often"];
    const expectedTiers = ["Much less", "Somewhat less", "About the same", "Somewhat more", "Much more"];

    const modes = [
      "Private Vehicle",
      "Ridehail/Taxi",
      "Transit",
      "Personal Bike",
      "Shared Bike",
      "Walking"
    ];

    const x = d3.scaleBand()
      .domain(modes)
      .range([0, width])
      .padding(0.2);

    const y = d3.scaleLinear()
      .domain([0, 100])
      .range([height, 0]);

    const usageColor = d3.scaleOrdinal()
      .domain(usageTiers)
      .range(["#d9d9d9", "#9ecae1", "#fdae6b", "#e6550d"]);

    const expectedColor = d3.scaleOrdinal()
      .domain(expectedTiers)
      .range(["#2166ac", "#92c5de", "#f7f7f7", "#f4a582", "#ca0020"]);

    // Axes
    g.append("g")
      .attr("transform", `translate(0, ${{height}})`)
      .call(d3.axisBottom(x))
      .selectAll("text")
      .attr("transform", "rotate(-25)")
      .style("text-anchor", "end")
      .style("font-family", "Georgia")
      .style("font-size", "12px");

    g.append("g")
      .call(d3.axisLeft(y))
      .selectAll("text")
      .style("font-family", "Georgia")
      .style("font-size", "12px");

    // Axis labels
    g.append("text")
      .attr("x", width / 2)
      .attr("y", height + 72)
      .attr("text-anchor", "middle")
      .style("font-family", "Georgia")
      .style("font-size", "12px")
      .text("Transportation Mode");

    g.append("text")
      .attr("transform", "rotate(-90)")
      .attr("x", -height / 2)
      .attr("y", -55)
      .attr("text-anchor", "middle")
      .style("font-family", "Georgia")
      .style("font-size", "12px")
      .text("Percent of respondents (%)");

    // Legend group
    const legend = svg.append("g")
      .attr("transform", `translate(${{width + margin.left + 25}}, ${{margin.top + 20}})`);

    function drawLegend(period) {{
      legend.selectAll("*").remove();
      const tiers      = period === "Expected After" ? expectedTiers : usageTiers;
      const colorScale = period === "Expected After" ? expectedColor : usageColor;

      tiers.forEach((tier, i) => {{
        const row = legend.append("g")
          .attr("transform", `translate(0, ${{i * 28}})`);
        row.append("rect")
          .attr("width", 16).attr("height", 16)
          .attr("fill", colorScale(tier))
          .attr("stroke", "#999");
        row.append("text")
          .attr("x", 26).attr("y", 12)
          .style("font-family", "Georgia")
          .style("font-size", "12px")
          .text(tier);
      }});
    }}

    function prepareData(period) {{
      const filtered = rawData.filter(d => d.period === period);
      const tiers    = period === "Expected After" ? expectedTiers : usageTiers;

      const modeMap = new Map();
      modes.forEach(mode => {{
        modeMap.set(mode, {{ mode }});
        tiers.forEach(tier => {{ modeMap.get(mode)[tier] = 0; }});
      }});

      filtered.forEach(d => {{ modeMap.get(d.mode)[d.tier] = +d.percent; }});
      return {{ data: Array.from(modeMap.values()), tiers }};
    }}

    function updateChart(period, clickedBtn) {{
      // Toggle active button style
      document.querySelectorAll(".buttons button").forEach(b => b.classList.remove("active"));
      if (clickedBtn) clickedBtn.classList.add("active");

      // Show/hide scale warning
      document.getElementById("scaleWarning").style.display =
        period === "Expected After" ? "block" : "none";

      const prepared   = prepareData(period);
      const data       = prepared.data;
      const tiers      = prepared.tiers;
      const colorScale = period === "Expected After" ? expectedColor : usageColor;

      const stacked = d3.stack().keys(tiers)(data);

      drawLegend(period);

      // Period label
      svg.selectAll(".period-label").remove();
      svg.append("text")
        .attr("class", "period-label")
        .attr("x", margin.left)
        .attr("y", 18)
        .style("font-family", "Georgia")
        .style("font-size", "13px")
        .style("font-weight", "bold")
        .text(`Period shown: ${{period}}`);

      // Bars
      const layers = g.selectAll(".layer")
        .data(stacked, d => d.key);

      layers.exit().remove();

      const layerMerge = layers.enter()
        .append("g")
        .attr("class", "layer")
        .merge(layers)
        .attr("fill", d => colorScale(d.key));

      const rects = layerMerge.selectAll("rect")
        .data(d => d.map(v => ({{ ...v, key: d.key }})), d => d.data.mode);

      rects.exit().remove();

      rects.enter()
        .append("rect")
        .attr("x", d => x(d.data.mode))
        .attr("width", x.bandwidth())
        .attr("y", height)
        .attr("height", 0)
        .on("mousemove", function(event, d) {{
          tooltip
            .style("opacity", 1)
            .html(
              `<strong>${{d.data.mode}}</strong><br>` +
              `${{d.key}}: ${{(d[1] - d[0]).toFixed(1)}}%`
            )
            .style("left", `${{event.pageX + 12}}px`)
            .style("top",  `${{event.pageY - 28}}px`);
        }})
        .on("mouseout", () => tooltip.style("opacity", 0))
        .merge(rects)
        .transition().duration(800)
        .attr("x",      d => x(d.data.mode))
        .attr("width",  x.bandwidth())
        .attr("y",      d => y(d[1]))
        .attr("height", d => y(d[0]) - y(d[1]));
    }}

    // Initialize
    updateChart("Before COVID", document.getElementById("btn-before"));
  </script>
</body>
</html>
"""

    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Saved HTML: {output_path}")
    return output_path


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_clean_data()

    # Quick encoding check — remove after confirming values are 0–4
    print("Pre-column value check:")
    print(df["tr_freq_pre_privateveh"].value_counts().sort_index())

    # Viz 2
    fig2 = vis2_mode_shift_heatmap(df, group_col="income_bracket")
    vis2_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vis2.png")
    fig2.savefig(vis2_path, dpi=300, bbox_inches="tight")
    print(f"Saved VIS 2: {vis2_path}")

    # Viz 5
    vis5_data = build_vis5_data(df)
    vis5_path = export_vis5_html(vis5_data, output_filename="vis5.html")

    webbrowser.open("file://" + os.path.abspath(vis2_path))
    webbrowser.open("file://" + os.path.abspath(vis5_path))
    print("Both visualizations generated and opened.")