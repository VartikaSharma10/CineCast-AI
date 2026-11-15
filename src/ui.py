"""
CineCast AI - UI Components and Layout Helpers
Provides consistent header branding, custom styling injection, KPI metric cards,
and unified sidebar design across all Streamlit application pages.
"""

import streamlit as st
from pathlib import Path
from src.config import COLORS, ASSETS_DIR


def inject_custom_css():
    """
    Reads assets/style.css and injects it into the Streamlit app.
    """
    css_path = ASSETS_DIR / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def init_page(page_title: str, icon: str = "🎬"):
    """
    Standardizes page configuration, title, layout, and style injection.
    """
    st.set_page_config(
        page_title=f"{page_title} | CineCast AI",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded"
    )
    inject_custom_css()
    render_sidebar_branding()


def render_sidebar_branding():
    """
    Renders luxury enterprise cinema branding on the sidebar with crisp high contrast text.
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 0.5rem 0 1.2rem 0; border-bottom: 1px solid #283C37; margin-bottom: 1.2rem;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="background-color: #8C6B2B; width: 34px; height: 34px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-weight: bold; color: #FFFFFF; font-size: 1.15rem; font-family: serif; border: 1px solid #DFCDB0;">
                        C
                    </div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 700; color: #FFFFFF; letter-spacing: -0.01em; line-height: 1.1;">
                            CineCast <span style="color: #DFCDB0;">AI</span>
                        </div>
                        <div style="font-size: 0.72rem; color: #C2D6CF; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">
                            Cinema Intelligence Suite
                        </div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def render_page_header(title: str, subtitle: str, badge_text: str = "ENTERPRISE BI"):
    """
    Renders luxury page header with title, subtitle, and badge in high contrast.
    """
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;">
                <div>
                    <h1 class="page-title">{title}</h1>
                    <div class="page-subtitle">{subtitle}</div>
                </div>
                <div>
                    <span class="badge-gold">{badge_text}</span>
                </div>
            </div>
            <div class="gold-divider"></div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_kpi_card(label: str, value: str, subtext: str = "", is_gold: bool = False):
    """
    HTML KPI metric card with consistent borders and high-contrast typography.
    """
    val_class = "kpi-value-gold" if is_gold else "kpi-value"
    subtext_html = f'<div class="kpi-subtext">{subtext}</div>' if subtext else ""
    return f"""
    <div class="kpi-container">
        <div>
            <div class="kpi-label">{label}</div>
            <div class="{val_class}">{value}</div>
        </div>
        {subtext_html}
    </div>
    """


def render_synthetic_data_banner():
    """
    Transparently notes that data is synthetically calibrated for technical assessment with high-contrast text.
    """
    st.markdown(
        """
        <div style="background-color: #EAE5D8; border: 1px solid #D5CEBF; border-radius: 6px; padding: 0.75rem 1rem; font-size: 0.82rem; color: #1E2E2A; display: flex; align-items: center; justify-content: space-between; margin-bottom: 1.5rem;">
            <div>
                <strong style="color: #172321;">Simulation Environment:</strong> Operating on 6,500+ calibrated synthetic screening records with weekend surges, release decay curves, and price elasticity.
            </div>
            <span class="badge-sage">CALIBRATED DATASET</span>
        </div>
        """,
        unsafe_allow_html=True
    )
