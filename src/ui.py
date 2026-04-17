"""
Shared UI helpers — call inject_sidebar() at the top of every page
so the logo, CSS, and font sizing are consistent across all pages.
"""

import os
import base64
import streamlit as st

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")


def _b64(filename: str) -> str:
    with open(os.path.join(_ASSETS, filename), "rb") as f:
        return base64.b64encode(f.read()).decode()


def inject_sidebar() -> None:
    """Inject global CSS and sidebar logo. Call once near the top of each page."""
    nav_b64 = _b64("beatthestreet_nav_icon.png")

    st.markdown(
        f"""
        <style>
        /* Global +2px font size (Streamlit base 16px → 18px) */
        html {{ font-size: 18px !important; }}

        /* Reorder sidebar: logo above the auto-generated nav list */
        [data-testid="stSidebarContent"] {{
            display: flex;
            flex-direction: column;
        }}
        [data-testid="stSidebarNav"] {{
            order: 2;
        }}
        [data-testid="stSidebarContent"] > div:not([data-testid="stSidebarNav"]) {{
            order: 1;
        }}

        /* Larger nav link text */
        [data-testid="stSidebarNav"] a span,
        [data-testid="stSidebarNav"] a p {{
            font-size: 1.2rem !important;
        }}
        [data-testid="stSidebarNav"] a {{
            padding: 10px 16px !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.sidebar.markdown(
        f"""
        <div style="display:flex; justify-content:center; padding: 16px 0 20px 0;">
            <img src="data:image/png;base64,{nav_b64}" width="150">
        </div>
        """,
        unsafe_allow_html=True,
    )
