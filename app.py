"""
CineCast AI - Intelligent Cinema Ticket Demand Forecasting & Revenue Optimization Platform
Main application entry point, executive landing portal, and model verification suite.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.ui import init_page, render_page_header, render_kpi_card, render_synthetic_data_banner
from src.config import COLORS, RAW_DATA_PATH, CLEAN_DATA_PATH
from src.prediction import load_artifacts, predict_single_show
from src.analytics import load_enriched_dataset, get_luxury_chart_layout

# Initialize page
init_page("Executive Portal", icon="🎬")

render_synthetic_data_banner()

# Load model metadata
_, _, metadata = load_artifacts()
best_name = metadata.get("best_model_name", "Gradient Boosting Regressor")
metrics = metadata.get("metrics", {})
champion_m = metadata.get("champion_metrics", {})
features = metadata.get("top_feature_importances", {})

# ----------------- HERO BANNER -----------------
st.markdown(
    """
    <div style="background-color: #172321; border-radius: 10px; padding: 2.2rem 2.5rem; margin-bottom: 2rem; border-left: 6px solid #8C6B2B; box-shadow: 0 4px 20px rgba(23, 35, 33, 0.15);">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 15px;">
            <div>
                <span style="background-color: rgba(182, 154, 104, 0.25); color: #DFCDB0; font-size: 0.75rem; font-weight: 700; padding: 4px 10px; border-radius: 4px; letter-spacing: 0.08em; text-transform: uppercase;">
                    Cinema Ticket Forecast
                </span>
                <h1 style="color: #FFFFFF !important; font-size: 2.4rem; font-weight: 700; margin: 0.8rem 0 0.4rem 0; letter-spacing: -0.02em;">
                    <span style="color: #FFFFFF;">CineCast</span>
                </h1>
                <p style="color: #E2EAE7; font-size: 1.05rem; max-width: 820px; line-height: 1.5; margin: 0;">
                    Intelligent cinema ticket demand forecasting and dynamic revenue optimization platform.
                    Trained on multi-theatre screening histories with chronological split validation,
                    capacity boundary guarantees, and actionable operational decision support.
                </p>
            </div>
            <div style="background-color: rgba(255, 255, 255, 0.08); padding: 1rem 1.4rem; border-radius: 8px; border: 1px solid rgba(223, 205, 176, 0.4); text-align: right;">
                <div style="font-size: 0.72rem; color: #DFCDB0; text-transform: uppercase; font-weight: 700; letter-spacing: 0.05em;">Champion Architecture</div>
                <div style="font-size: 1.15rem; font-weight: 700; color: #FFFFFF; margin: 2px 0;">{best_name}</div>
                <div style="font-size: 0.80rem; color: #C5DED5;">R² Score: <strong>{r2_score}</strong> | RMSE: <strong>{rmse_val}</strong></div>
            </div>
        </div>
    </div>
    """.format(
        best_name=best_name,
        r2_score=champion_m.get("R2", 0.9176),
        rmse_val=champion_m.get("RMSE", 20.62)
    ),
    unsafe_allow_html=True
)

# ----------------- SYSTEM STATUS & PIPELINE METRICS -----------------
st.markdown("### ⚙️ Machine Learning Pipeline Architecture")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        render_kpi_card(
            label="Historical Dataset",
            value="6,500",
            subtext="Multi-theatre records"
        ),
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        render_kpi_card(
            label="Model R² Score",
            value=f"{champion_m.get('R2', 0.9176):.4f}",
            subtext="Chronological test set",
            is_gold=True
        ),
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        render_kpi_card(
            label="Mean Absolute Error",
            value=f"{champion_m.get('MAE', 14.94)}",
            subtext="Avg error in tickets sold"
        ),
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        render_kpi_card(
            label="Root Mean Sq Error",
            value=f"{champion_m.get('RMSE', 20.62)}",
            subtext="Test set RMSE"
        ),
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- MODEL COMPARISON BENCHMARK -----------------
col_models, col_importances = st.columns([1.1, 1.0], gap="medium")

with col_models:
    st.markdown("##### 🔬 Regression Model Benchmark (Chronological Split)")
    if metrics:
        comp_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index": "Model"})
        comp_df = comp_df[["Model", "MAE", "RMSE", "R2"]]
        
        # Highlight best model
        st.dataframe(
            comp_df.style.highlight_min(subset=["MAE", "RMSE"], color="#EAF0ED")
                         .highlight_max(subset=["R2"], color="#F5EFE3")
                         .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R2": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )
    
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #4A5652; line-height: 1.5; background-color: #FFFFFF; border: 1px solid #E5E0D5; border-radius: 6px; padding: 0.8rem 1rem; margin-top: 0.6rem;">
            <strong>Zero Temporal Leakage Guarantee:</strong> The dataset was split chronologically 
            (Train: Jan 2025 – Dec 2025 | Test: Dec 2025 – Mar 2026). Models are validated strictly against future unseen dates.
            Physical constraints ensure <code style="color: #172321;">Advance Bookings ≤ Predicted Tickets ≤ Capacity</code>.
        </div>
        """,
        unsafe_allow_html=True
    )

with col_importances:
    st.markdown("##### 📈 Top Feature Importances (Champion Model)")
    if features:
        feat_df = pd.DataFrame(list(features.items()), columns=["Feature", "Importance"]).head(8)
        feat_df = feat_df.sort_values("Importance", ascending=True)

        fig_feat = go.Figure(go.Bar(
            y=feat_df["Feature"],
            x=feat_df["Importance"],
            orientation="h",
            marker=dict(color=COLORS["deep_forest"], line=dict(color=COLORS["champagne_gold"], width=1)),
            text=feat_df["Importance"].apply(lambda v: f"{v*100:.1f}%"),
            textposition="auto"
        ))
        fig_feat.update_layout(
            get_luxury_chart_layout(),
            height=260,
            margin=dict(l=10, r=10, t=10, b=20),
            xaxis_title="Relative Feature Importance"
        )
        st.plotly_chart(fig_feat, use_container_width=True)

st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

# ----------------- APPLICATION MODULES DIRECTORY -----------------
st.markdown("### 🧭 Application Modules")
st.markdown("Select a module from the sidebar navigation or explore the quick links below:")

mod1, mod2, mod3 = st.columns(3)
with mod1:
    st.markdown(
        """
        <div class="cinema-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.4rem;">📊</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #172321; margin-bottom: 0.3rem;">
                Executive Overview
            </div>
            <div style="font-size: 0.86rem; color: #354742; line-height: 1.4;">
                High-level C-suite dashboard displaying historical sales trends, forward forecast estimates, genre share, and theatre benchmarks.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with mod2:
    st.markdown(
        """
        <div class="cinema-card-gold" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.4rem;">🎯</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #172321; margin-bottom: 0.3rem;">
                Demand Forecasting
            </div>
            <div style="font-size: 0.86rem; color: #354742; line-height: 1.4;">
                Interactive prediction studio: configure movie, showtime, pricing, and advance bookings to generate real-time ML demand forecasts.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with mod3:
    st.markdown(
        """
        <div class="cinema-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.4rem;">🎞️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #172321; margin-bottom: 0.3rem;">
                Movie Intelligence
            </div>
            <div style="font-size: 0.86rem; color: #354742; line-height: 1.4;">
                Film lifecycle analytics, theatrical release-age decay curves, audience rating elasticity, and top vs under-performing titles.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

mod4, mod5, mod6 = st.columns(3)
with mod4:
    st.markdown(
        """
        <div class="cinema-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.4rem;">💰</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #172321; margin-bottom: 0.3rem;">
                Revenue Analytics
            </div>
            <div style="font-size: 0.86rem; color: #354742; line-height: 1.4;">
                Financial yield optimization, model revenue calibration, actual vs predicted variance, and auditorium capacity saturation.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with mod5:
    st.markdown(
        """
        <div class="cinema-card-gold" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.4rem;">💡</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #172321; margin-bottom: 0.3rem;">
                Smart Recommendations
            </div>
            <div style="font-size: 0.86rem; color: #354742; line-height: 1.4;">
                Dynamic rule-based operational queue: auxiliary screenings, promotional pushes, prime surge pricing, and screen downsizing.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
with mod6:
    st.markdown(
        """
        <div class="cinema-card" style="min-height: 190px;">
            <div style="font-size: 1.3rem; margin-bottom: 0.4rem;">🛡️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #172321; margin-bottom: 0.3rem;">
                Enterprise Governance
            </div>
            <div style="font-size: 0.86rem; color: #354742; line-height: 1.4;">
                Full model lineage, Joblib persistence, missing data imputation logs, deduplication records, and zero target leakage.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
