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

    st.sidebar.markdown(
        """
        <div style="position:fixed; bottom:1.5rem; left:0; width:var(--sidebar-width, 18rem);
                    display:flex; justify-content:center;">
            <a href="https://github.com/MikaIsmayilov/BeatTheStreet" target="_blank"
               style="display:flex; align-items:center; gap:6px; text-decoration:none;
                      color:#888; font-size:0.78rem;">
                <svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24"
                     fill="currentColor">
                  <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385
                           .6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235
                           -3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695
                           -.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23
                           1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605
                           -2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225
                           -.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23
                           .96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23
                           3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225
                           0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22
                           0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57
                           A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/>
                </svg>
                View on GitHub
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
