"""
CineCast AI - Page 2: Demand Forecasting Studio
Interactive prediction interface driven directly by the trained ML champion model.
Predicts ticket sales, occupancy %, expected revenue, remaining seats, and demand tier.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, date, timedelta

from src.ui import init_page, render_page_header, render_kpi_card, render_synthetic_data_banner
from src.config import MOVIES, THEATRES, SHOWTIMES, COLORS
from src.prediction import predict_single_show, load_artifacts

# Initialize page
init_page("Demand Forecasting", icon="🎯")

render_page_header(
    title="Demand Forecasting Studio",
    subtitle="Interactive ML inference engine for upcoming movie screenings and capacity planning",
    badge_text="LIVE MODEL INFERENCE"
)

# Load model metadata for display
_, _, metadata = load_artifacts()
best_model_name = metadata.get("best_model_name", "Gradient Boosting Regressor")
model_r2 = metadata.get("champion_metrics", {}).get("R2", 0.9176)
model_rmse = metadata.get("champion_metrics", {}).get("RMSE", 20.62)

# Quick scenario presets
st.markdown("##### ⚡ Quick Scenario Presets")
preset_col1, preset_col2, preset_col3, preset_col4 = st.columns(4)

preset_selected = None
with preset_col1:
    if st.button("🔥 Blockbuster Opening Night", use_container_width=True):
        st.session_state["preset"] = {
            "movie": "Dune: Prophecy",
            "theatre": "Grand Regal Cinema - Screen 1 (IMAX)",
            "time": "20:30",
            "price": 25.5,
            "adv": 260,
            "days": 1,
            "is_holiday": False
        }
with preset_col2:
    if st.button("☕ Weekday Morning Matinee", use_container_width=True):
        st.session_state["preset"] = {
            "movie": "The Grand Royale",
            "theatre": "Apex Luxe Multiplex - Screen 4",
            "time": "10:00",
            "price": 14.0,
            "adv": 25,
            "days": 18,
            "is_holiday": False
        }
with preset_col3:
    if st.button("🍿 Weekend Family Matinee", use_container_width=True):
        st.session_state["preset"] = {
            "movie": "The Cosmic Voyage",
            "theatre": "Starlight Dome - Grand Auditorium",
            "time": "13:30",
            "price": 18.0,
            "adv": 180,
            "days": 5,
            "is_holiday": False
        }
with preset_col4:
    if st.button("🌙 Late Night Horror Screening", use_container_width=True):
        st.session_state["preset"] = {
            "movie": "Silent Echoes",
            "theatre": "Velvet Screen Studios - Studio A",
            "time": "23:15",
            "price": 21.0,
            "adv": 55,
            "days": 7,
            "is_holiday": False
        }

active_preset = st.session_state.get("preset", {})

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- MAIN PREDICTION LAYOUT -----------------
form_col, result_col = st.columns([1.1, 1.3], gap="large")

with form_col:
    st.markdown("### 🎬 Screening Configuration")

    with st.container(border=True):
        # Movie Selection
        movie_titles = [m["title"] for m in MOVIES]
        default_movie_idx = 0
        if "movie" in active_preset and active_preset["movie"] in movie_titles:
            default_movie_idx = movie_titles.index(active_preset["movie"])

        selected_movie = st.selectbox(
            "Movie Title",
            options=movie_titles,
            index=default_movie_idx
        )

        movie_meta = next(m for m in MOVIES if m["title"] == selected_movie)

        # Genre & Rating (Auto-derived from movie, editable)
        c_gen, c_rate = st.columns(2)
        with c_gen:
            genre_list = sorted(list({m["genre"] for m in MOVIES}))
            selected_genre = st.selectbox(
                "Genre",
                options=genre_list,
                index=genre_list.index(movie_meta["genre"])
            )
        with c_rate:
            movie_rating = st.slider(
                "Audience Rating",
                min_value=5.0,
                max_value=10.0,
                value=float(movie_meta["base_rating"]),
                step=0.1
            )

        # Theatre Selection
        theatre_names = [t["name"] for t in THEATRES]
        default_theatre_idx = 0
        if "theatre" in active_preset and active_preset["theatre"] in theatre_names:
            default_theatre_idx = theatre_names.index(active_preset["theatre"])

        selected_theatre = st.selectbox(
            "Auditorium / Theatre",
            options=theatre_names,
            index=default_theatre_idx
        )

        theatre_meta = next(t for t in THEATRES if t["name"] == selected_theatre)

        # Capacity & Ticket Price
        c_cap, c_pr = st.columns(2)
        with c_cap:
            theatre_capacity = st.number_input(
                "Auditorium Capacity",
                min_value=50,
                max_value=600,
                value=int(theatre_meta["base_capacity"]),
                step=10
            )
        with c_pr:
            default_price = active_preset.get("price", 18.5)
            ticket_price = st.number_input(
                "Ticket Price ($)",
                min_value=8.0,
                max_value=50.0,
                value=float(default_price),
                step=0.5
            )

        # Date, Showtime, and Calendar Status
        c_date, c_time = st.columns(2)
        with c_date:
            show_date = st.date_input(
                "Show Date",
                value=date.today() + timedelta(days=2)
            )
        with c_time:
            time_options = [s["time"] for s in SHOWTIMES]
            time_labels = {s["time"]: s["label"] for s in SHOWTIMES}
            default_time_idx = 3 # Prime 20:30
            if "time" in active_preset and active_preset["time"] in time_options:
                default_time_idx = time_options.index(active_preset["time"])

            selected_time = st.selectbox(
                "Showtime",
                options=time_options,
                format_func=lambda t: time_labels.get(t, t),
                index=default_time_idx
            )

        # Advance Bookings & Release Age
        c_adv, c_days = st.columns(2)
        with c_adv:
            max_adv = int(theatre_capacity)
            default_adv = min(active_preset.get("adv", int(theatre_capacity * 0.35)), max_adv)
            advance_bookings = st.number_input(
                "Advance Bookings",
                min_value=0,
                max_value=max_adv,
                value=default_adv,
                step=5
            )
        with c_days:
            days_since_release = st.number_input(
                "Days Since Release",
                min_value=0,
                max_value=90,
                value=active_preset.get("days", 4),
                step=1
            )

        # Holiday status checkbox
        is_holiday = st.checkbox(
            "National Holiday / Special Screening",
            value=active_preset.get("is_holiday", False)
        )

        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("⚡ Predict Ticket Demand", use_container_width=True)


with result_col:
    st.markdown("### 📊 Predictive Intelligence")

    # Run prediction automatically on first load or button click
    show_inputs = {
        "movie_title": selected_movie,
        "genre": selected_genre,
        "theatre_name": selected_theatre,
        "show_date": show_date,
        "show_time": selected_time,
        "ticket_price": ticket_price,
        "theatre_capacity": theatre_capacity,
        "advance_bookings": advance_bookings,
        "movie_rating": movie_rating,
        "days_since_release": days_since_release,
        "is_holiday": 1 if is_holiday else 0
    }

    forecast = predict_single_show(show_inputs)

    # 4 Output Metric Cards
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""
            <div class="kpi-container" style="border-top: 3px solid #172321;">
                <div>
                    <div class="kpi-label">Predicted Ticket Sales</div>
                    <div class="kpi-value">{forecast['predicted_tickets']:,} <span style="font-size: 1rem; color: #7D8885; font-weight: 500;">/ {theatre_capacity}</span></div>
                </div>
                <div class="kpi-subtext">Est. Interval: {forecast['confidence_lower']} – {forecast['confidence_upper']} seats</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        badge = forecast["demand_badge"]
        st.markdown(
            f"""
            <div class="kpi-container" style="border-top: 3px solid {badge['badge_color']};">
                <div>
                    <div class="kpi-label">Expected Occupancy</div>
                    <div class="kpi-value-gold">{forecast['occupancy_pct']}%</div>
                </div>
                <div style="margin-top: 0.4rem;">
                    <span style="background-color: {badge['badge_color']}; color: {badge['text_color']}; padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 700; text-transform: uppercase;">
                        {forecast['demand_tier']} Demand
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    m3, m4 = st.columns(2)
    with m3:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div>
                    <div class="kpi-label">Expected Ticket Revenue</div>
                    <div class="kpi-value" style="color: #172321;">${forecast['expected_revenue']:,.2f}</div>
                </div>
                <div class="kpi-subtext">At ${ticket_price:.2f} per ticket</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m4:
        st.markdown(
            f"""
            <div class="kpi-container">
                <div>
                    <div class="kpi-label">Remaining Available Seats</div>
                    <div class="kpi-value">{forecast['remaining_seats']:,}</div>
                </div>
                <div class="kpi-subtext">{theatre_capacity - forecast['predicted_tickets']} unallocated seats</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualization: Gauge and Breakdown Donut
    tab_gauge, tab_breakdown = st.tabs(["Auditorium Occupancy Gauge", "Seat Allocation Breakdown"])

    with tab_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=forecast["occupancy_pct"],
            number={"suffix": "%", "font": {"color": COLORS["deep_forest"], "size": 36, "family": "Plus Jakarta Sans"}},
            title={"text": "<b>Projected Seat Capacity Fill Rate</b>", "font": {"size": 14, "color": COLORS["charcoal"]}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": COLORS["muted_gray"]},
                "bar": {"color": COLORS["deep_forest"], "thickness": 0.28},
                "bgcolor": "white",
                "borderwidth": 1,
                "bordercolor": "#E5E0D5",
                "steps": [
                    {"range": [0, 40], "color": "#F9ECEC"},     # Low (Soft red)
                    {"range": [40, 65], "color": "#F7F2E7"},    # Moderate (Soft amber)
                    {"range": [65, 85], "color": "#EAF0ED"},    # High (Soft sage)
                    {"range": [85, 100], "color": "#F5EFE3"}    # Very High (Soft gold)
                ],
                "threshold": {
                    "line": {"color": COLORS["champagne_gold"], "width": 4},
                    "thickness": 0.8,
                    "value": forecast["occupancy_pct"]
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with tab_breakdown:
        walk_in_est = max(0, forecast["predicted_tickets"] - advance_bookings)
        unsold_est = forecast["remaining_seats"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=["Advance Bookings", "Projected Walk-in Sales", "Unsold Capacity"],
            values=[advance_bookings, walk_in_est, unsold_est],
            hole=0.6,
            marker=dict(colors=[COLORS["champagne_gold"], COLORS["sage_green"], "#E5E0D5"]),
            textinfo="label+value",
            textposition="inside"
        )])
        fig_donut.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # Dynamic Operational Recommendation Card
    st.markdown(
        f"""
        <div class="cinema-card-gold" style="margin-top: 1rem;">
            <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 0.5rem;">
                <span style="font-size: 0.78rem; font-weight: 700; color: #B69A68; text-transform: uppercase; letter-spacing: 0.05em;">
                    Operational Recommendation
                </span>
                <span class="badge-sage">ACTIVE MODEL: {best_model_name.upper()} (R²: {model_r2})</span>
            </div>
            <div style="font-size: 0.95rem; font-weight: 600; color: #172321; line-height: 1.4;">
                {forecast['quick_recommendation']}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
