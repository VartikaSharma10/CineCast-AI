"""
CineCast AI - Page 1: Executive Overview Dashboard
High-level cinema business intelligence dashboard displaying historical performance,
predicted forward bookings, occupancy benchmarks, and multi-theatre trends.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from src.ui import init_page, render_page_header, render_kpi_card, render_synthetic_data_banner
from src.analytics import (
    load_enriched_dataset,
    calculate_executive_kpis,
    create_actual_vs_predicted_trend_chart,
    create_genre_demand_chart,
    create_theatre_performance_chart,
    create_showtime_occupancy_heatmap
)

# Initialize standard styling & branding
init_page("Executive Overview", icon="📊")

render_page_header(
    title="Executive Overview",
    subtitle="Holistic performance metrics, forward booking estimates, and cross-theatre benchmarks",
    badge_text="C-SUITE BRIEFING"
)

render_synthetic_data_banner()

# Load data with caching
@st.cache_data(show_spinner="Loading cinema operational records...")
def get_data():
    return load_enriched_dataset()

df_full = get_data()

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Operational Filters")

    # Date Range Filter
    min_date = df_full["show_date_dt"].min().date()
    max_date = df_full["show_date_dt"].max().date()
    date_range = st.date_input(
        "Date Range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    # Theatre MultiSelect
    all_theatres = sorted(df_full["theatre_name"].unique())
    selected_theatres = st.multiselect(
        "Auditiorium / Theatre",
        options=all_theatres,
        default=all_theatres
    )

    # Genre MultiSelect
    all_genres = sorted(df_full["genre"].unique())
    selected_genres = st.multiselect(
        "Movie Genre",
        options=all_genres,
        default=all_genres
    )

# Apply Filters
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = df_full[
        (df_full["show_date_dt"].dt.date >= start_d) &
        (df_full["show_date_dt"].dt.date <= end_d)
    ]
else:
    filtered_df = df_full.copy()

if selected_theatres:
    filtered_df = filtered_df[filtered_df["theatre_name"].isin(selected_theatres)]
if selected_genres:
    filtered_df = filtered_df[filtered_df["genre"].isin(selected_genres)]

if filtered_df.empty:
    st.warning("No screenings match the selected filter criteria. Please adjust your filters in the sidebar.")
    st.stop()

# ----------------- EXECUTIVE KPIS -----------------
kpis = calculate_executive_kpis(filtered_df)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(
        render_kpi_card(
            label="Historical Tickets Sold",
            value=f"{kpis['total_historical_tickets']:,}",
            subtext=f"Across {kpis['total_screenings']:,} shows"
        ),
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        render_kpi_card(
            label="Predicted Bookings",
            value=f"{kpis['predicted_upcoming_bookings']:,}",
            subtext="ML forward forecast",
            is_gold=True
        ),
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        render_kpi_card(
            label="Expected Revenue",
            value=f"${kpis['expected_ticket_revenue']:,.0f}",
            subtext=f"Historical: ${kpis['total_historical_revenue']:,.0f}"
        ),
        unsafe_allow_html=True
    )

with col4:
    st.markdown(
        render_kpi_card(
            label="Avg Theatre Occupancy",
            value=f"{kpis['average_theatre_occupancy']}%",
            subtext="Industry target: 55-65%"
        ),
        unsafe_allow_html=True
    )

with col5:
    st.markdown(
        render_kpi_card(
            label="Highest-Demand Movie",
            value=kpis['highest_demand_movie'].split(":")[0],
            subtext=f"~{kpis['highest_demand_movie_avg']} avg tickets/show"
        ),
        unsafe_allow_html=True
    )

with col6:
    st.markdown(
        render_kpi_card(
            label="Peak Showtime",
            value=kpis['peak_demand_showtime'],
            subtext="Prime evening window"
        ),
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- CHARTS ROW 1: TIME SERIES & GENRE -----------------
col_chart_left, col_chart_right = st.columns([1.5, 1.0])

with col_chart_left:
    st.plotly_chart(
        create_actual_vs_predicted_trend_chart(filtered_df),
        use_container_width=True
    )

with col_chart_right:
    st.plotly_chart(
        create_genre_demand_chart(filtered_df),
        use_container_width=True
    )

# ----------------- CHARTS ROW 2: THEATRE & HEATMAP -----------------
col_theatre, col_heatmap = st.columns([1.2, 1.0])

with col_theatre:
    st.plotly_chart(
        create_theatre_performance_chart(filtered_df),
        use_container_width=True
    )

with col_heatmap:
    st.plotly_chart(
        create_showtime_occupancy_heatmap(filtered_df),
        use_container_width=True
    )

# ----------------- UPCOMING DEMAND SUMMARY -----------------
st.markdown("### 📋 Upcoming Screenings Demand Summary")

display_cols = [
    "show_date", "show_time", "movie_title", "theatre_name",
    "theatre_capacity", "advance_bookings", "predicted_tickets",
    "predicted_occupancy", "demand_category", "predicted_revenue"
]

summary_table = filtered_df[display_cols].sort_values("show_date", ascending=False).head(15).copy()
summary_table.columns = [
    "Date", "Showtime", "Movie Title", "Auditorium",
    "Capacity", "Advance", "Pred Tickets",
    "Occupancy %", "Demand Tier", "Est Revenue ($)"
]

st.dataframe(
    summary_table.style.format({
        "Capacity": "{:,}",
        "Advance": "{:,}",
        "Pred Tickets": "{:,}",
        "Occupancy %": "{:.1f}%",
        "Est Revenue ($)": "${:,.2f}"
    }),
    use_container_width=True,
    hide_index=True
)
