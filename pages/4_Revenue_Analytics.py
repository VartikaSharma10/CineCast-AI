"""
CineCast AI - Page 4: Revenue & Occupancy Analytics
Comprehensive financial yield analysis, actual vs predicted revenue variance,
capacity saturation, and temporal box office trends.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui import init_page, render_page_header, render_kpi_card, render_synthetic_data_banner
from src.config import COLORS
from src.analytics import (
    load_enriched_dataset,
    get_luxury_chart_layout,
    create_revenue_comparison_chart,
    create_showtime_occupancy_heatmap
)

# Initialize page
init_page("Revenue Analytics", icon="💰")

render_page_header(
    title="Revenue & Occupancy Analytics",
    subtitle="Financial yield optimization, model revenue calibration, and capacity saturation benchmarks",
    badge_text="FINANCIAL BI"
)

render_synthetic_data_banner()

@st.cache_data(show_spinner="Aggregating revenue analytics...")
def get_data():
    return load_enriched_dataset()

df_full = get_data()

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Financial Filters")
    min_date = df_full["show_date_dt"].min().date()
    max_date = df_full["show_date_dt"].max().date()
    date_range = st.date_input("Date Range", value=(min_date, max_date))

    all_theatres = sorted(df_full["theatre_name"].unique())
    selected_theatres = st.multiselect("Auditorium", options=all_theatres, default=all_theatres)

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    filtered_df = df_full[
        (df_full["show_date_dt"].dt.date >= start_d) &
        (df_full["show_date_dt"].dt.date <= end_d) &
        (df_full["theatre_name"].isin(selected_theatres))
    ]
else:
    filtered_df = df_full[df_full["theatre_name"].isin(selected_theatres)]

if filtered_df.empty:
    st.warning("No revenue records match the selected criteria.")
    st.stop()

# ----------------- FINANCIAL KPIS -----------------
tot_actual_rev = filtered_df["actual_revenue"].sum()
tot_pred_rev = filtered_df["predicted_revenue"].sum()
rev_diff = tot_actual_rev - tot_pred_rev
accuracy_ratio = 100.0 - abs(rev_diff / tot_actual_rev * 100.0) if tot_actual_rev > 0 else 0
avg_ticket_price = filtered_df["ticket_price"].mean()

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        render_kpi_card(
            label="Gross Actual Revenue",
            value=f"${tot_actual_rev:,.0f}",
            subtext=f"{filtered_df['booking_id'].count():,} Total Screenings"
        ),
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        render_kpi_card(
            label="Predicted Revenue",
            value=f"${tot_pred_rev:,.0f}",
            subtext="ML forward model projection",
            is_gold=True
        ),
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        render_kpi_card(
            label="Revenue Calibration",
            value=f"{accuracy_ratio:.2f}%",
            subtext=f"Variance: ${rev_diff:+,.0f}"
        ),
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        render_kpi_card(
            label="Avg Ticket Price",
            value=f"${avg_ticket_price:.2f}",
            subtext="Portfolio-weighted yield"
        ),
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- CHARTS ROW 1: ACTUAL VS PREDICTED BY MOVIE -----------------
st.plotly_chart(
    create_revenue_comparison_chart(filtered_df),
    use_container_width=True
)

# ----------------- CHARTS ROW 2: MONTHLY TREND & CAPACITY SATURATION -----------------
col_trend, col_capacity = st.columns([1.2, 1.0])

with col_trend:
    # Monthly aggregate
    df_trend = filtered_df.copy()
    df_trend["month_period"] = df_trend["show_date_dt"].dt.to_period("M").astype(str)
    monthly_rev = df_trend.groupby("month_period").agg(
        actual_rev=("actual_revenue", "sum"),
        pred_rev=("predicted_revenue", "sum")
    ).reset_index()

    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=monthly_rev["month_period"],
        y=monthly_rev["actual_rev"],
        name="Actual Revenue",
        marker=dict(color=COLORS["deep_forest"])
    ))
    fig_month.add_trace(go.Bar(
        x=monthly_rev["month_period"],
        y=monthly_rev["pred_rev"],
        name="Predicted Revenue",
        marker=dict(color=COLORS["champagne_gold"])
    ))
    fig_month.update_layout(
        get_luxury_chart_layout(),
        title="<b>Monthly Box Office Trajectory ($)</b>",
        barmode="group",
        xaxis_title="Billing Month",
        yaxis_title="Revenue ($)"
    )
    st.plotly_chart(fig_month, use_container_width=True)

with col_capacity:
    # Demand vs Capacity Saturation Scatter
    sample_df = filtered_df.sample(min(800, len(filtered_df)), random_state=42)
    fig_sat = go.Figure()

    # Scatter of individual screenings
    fig_sat.add_trace(go.Scatter(
        x=sample_df["theatre_capacity"],
        y=sample_df["tickets_sold"],
        mode="markers",
        marker=dict(
            color=sample_df["actual_occupancy"],
            colorscale=[[0, "#E5E0D5"], [0.5, "#6F8980"], [1.0, "#B69A68"]],
            size=7,
            opacity=0.7,
            colorbar=dict(title="Occ %")
        ),
        name="Screenings"
    ))

    # Diagonal 100% capacity saturation line
    max_cap = int(sample_df["theatre_capacity"].max())
    fig_sat.add_trace(go.Scatter(
        x=[0, max_cap],
        y=[0, max_cap],
        mode="lines",
        line=dict(color=COLORS["danger_red"], dash="dash", width=2),
        name="100% Capacity Ceiling"
    ))

    fig_sat.update_layout(
        get_luxury_chart_layout(),
        title="<b>Demand vs Auditorium Capacity Saturation</b>",
        xaxis_title="Total Seat Capacity",
        yaxis_title="Tickets Sold",
        showlegend=True
    )
    st.plotly_chart(fig_sat, use_container_width=True)

# ----------------- THEATRE LEVEL PERFORMANCE TABLE -----------------
st.markdown("### 🏢 Theatre Auditorium Financial Benchmarks")

theatre_table = filtered_df.groupby("theatre_name").agg(
    screenings=("booking_id", "count"),
    capacity=("theatre_capacity", "first"),
    avg_price=("ticket_price", "mean"),
    total_actual_rev=("actual_revenue", "sum"),
    total_pred_rev=("predicted_revenue", "sum"),
    avg_occupancy=("actual_occupancy", "mean")
).reset_index()

theatre_table["rev_variance"] = theatre_table["total_actual_rev"] - theatre_table["total_pred_rev"]
theatre_table = theatre_table.sort_values("total_actual_rev", ascending=False)

st.dataframe(
    theatre_table.rename(columns={
        "theatre_name": "Auditorium / Screen",
        "screenings": "Screenings",
        "capacity": "Seat Capacity",
        "avg_price": "Avg Ticket ($)",
        "total_actual_rev": "Actual Revenue ($)",
        "total_pred_rev": "Predicted Revenue ($)",
        "rev_variance": "Variance ($)",
        "avg_occupancy": "Avg Occupancy %"
    }).style.format({
        "Seat Capacity": "{:,}",
        "Screenings": "{:,}",
        "Avg Ticket ($)": "${:.2f}",
        "Actual Revenue ($)": "${:,.2f}",
        "Predicted Revenue ($)": "${:,.2f}",
        "Variance ($)": "${:+,.2f}",
        "Avg Occupancy %": "{:.1f}%"
    }),
    use_container_width=True,
    hide_index=True
)
