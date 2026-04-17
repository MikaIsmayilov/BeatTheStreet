"""
Home Page — BeatTheStreet landing page.
"""

import os
import re
import base64
import streamlit as st
import streamlit.components.v1 as _components

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")

# ── Header ────────────────────────────────────────────────────────────────────
with open(os.path.join(ASSETS, "beatthestreet_logo.svg"), "r") as _f:
    _svg = _f.read()


def _bake_svg(svg: str, dark: bool) -> str:
    """Replace <style> block with hardcoded theme colors and return base64."""
    if dark:
        wm, tl, dv, cr, cg = "#ffffff", "#00e396", "#333d4d", "#ff4d6a", "#00e396"
    else:
        wm, tl, dv, cr, cg = "#0a0e13", "#00a86b", "#cbd5e0", "#e03050", "#00a86b"
    style = (
        f"  <style>\n"
        f"    .wordmark {{ fill: {wm}; }}\n"
        f"    .tagline  {{ fill: {tl}; }}\n"
        f"    .divider  {{ stroke: {dv}; }}\n"
        f"    .c-red    {{ fill: {cr}; }}\n"
        f"    .w-red    {{ stroke: {cr}; }}\n"
        f"    .c-green  {{ fill: {cg}; }}\n"
        f"    .w-green  {{ stroke: {cg}; }}\n"
        f"  </style>"
    )
    themed = re.sub(r"<style>.*?</style>", style, svg, flags=re.DOTALL)
    return base64.b64encode(themed.encode()).decode()


_b64_light = _bake_svg(_svg, dark=False)
_b64_dark  = _bake_svg(_svg, dark=True)

# Detect the actual active Streamlit theme via JS reading the parent window's
# background color (st.get_option("theme.base") only reads config.toml, not
# the user's active theme selection). Falls back to prefers-color-scheme.
_components.html(
    f"""<!DOCTYPE html>
<html><head>
<style>
  html, body {{ margin:0; padding:0; overflow:hidden; }}
  img {{ max-width:780px; width:100%; display:block; }}
</style>
</head><body>
<img id="logo" src="data:image/svg+xml;base64,{_b64_light}">
<script>
(function() {{
  function applyTheme() {{
    var isDark = false;
    try {{
      var p  = window.parent.document;
      var el = p.querySelector('[data-testid="stApp"]') || p.body;
      var bg = window.parent.getComputedStyle(el).backgroundColor;
      var r  = parseInt((bg.match(/\d+/g) || [255])[0], 10);
      isDark = r < 50;
      document.body.style.background = isDark ? '#0e1117' : bg;
    }} catch(e) {{
      isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      document.body.style.background = isDark ? '#0e1117' : '#ffffff';
    }}
    document.getElementById('logo').src = isDark
      ? 'data:image/svg+xml;base64,{_b64_dark}'
      : 'data:image/svg+xml;base64,{_b64_light}';
  }}
  applyTheme();
  setTimeout(applyTheme, 80);
}})();
</script>
</body></html>""",
    height=165,
    scrolling=False,
)

st.markdown("""
Will a company **beat**, **meet**, or **miss** Wall Street's EPS estimates?
BeatTheStreet uses machine learning trained on 20 years of earnings data to generate
probability forecasts ahead of each announcement — combining analyst estimates,
company fundamentals, price momentum, and live macroeconomic signals.
""")

st.divider()

# ── Key stats ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Training Samples",   "104,938")
c2.metric("Model Accuracy",     "60.9%")
c3.metric("Years of History",   "2005 – 2024")
c4.metric("Predictive Features","27")
c5.metric("Best Model",         "LightGBM")

st.divider()

# ── Navigation guide ──────────────────────────────────────────────────────────
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### How it works")
    st.markdown("""
1. **Data** — 20 years of WRDS earnings data (Compustat + I/B/E/S + CRSP) forms
   the training set. Live inputs come from Yahoo Finance and FRED.
2. **Features** — Analyst consensus, estimate revisions, momentum signals,
   financial ratios, and macro indicators (VIX, oil, credit spreads, yields).
3. **Model** — LightGBM trained on a strict time-based split to prevent lookahead bias.
   Test set: 2022–2024 (never seen during training).
4. **Output** — Beat / Meet / Miss probabilities + SHAP chart explaining the drivers.
""")

with col_b:
    st.markdown("### Navigate")
    st.page_link("pages/1_Chart.py",              label="Price Chart",        help="Interactive candlestick chart")
    st.page_link("pages/1_Earnings_Predictor.py", label="Earnings Predictor", help="Enter a ticker for a live forecast")
    st.page_link("pages/2_Earnings_Calendar.py",  label="Earnings Calendar",  help="Upcoming earnings this week")
    st.page_link("pages/4_Sector_Overview.py",    label="Sector Overview",    help="Beat/miss rates by sector")
    st.page_link("pages/3_Backtesting.py",        label="Backtesting",        help="Historical model performance")

st.divider()
st.caption(
    "For informational and educational purposes only. Not financial advice. "
    "Predictions are probabilistic and may be incorrect — do not use as the sole basis for any investment decision."
)
