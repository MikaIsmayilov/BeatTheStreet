"""
BeatTheStreet — navigation shell.
Handles page config, shared sidebar, and routing for all pages.
"""

import os
import sys
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from ui import inject_sidebar

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

st.set_page_config(
    page_title="BeatTheStreet",
    page_icon=os.path.join(ASSETS, "beatthestreet_nav_icon.png"),
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_sidebar()

pg = st.navigation([
    st.Page("pages/0_Home.py",               title="Home",               default=True),
    st.Page("pages/1_Chart.py",              title="Price Chart"),
    st.Page("pages/1_Earnings_Predictor.py", title="Earnings Predictor"),
    st.Page("pages/2_Earnings_Calendar.py",  title="Earnings Calendar"),
    st.Page("pages/4_Sector_Overview.py",    title="Sector Overview"),
    st.Page("pages/3_Backtesting.py",        title="Backtesting"),
])
pg.run()
