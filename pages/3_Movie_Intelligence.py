"""
CineCast AI - Page 3: Movie Intelligence & Lifecycle Analytics
Movie-level performance deep-dive: title-wise sales, genre traction,
release-age decay curves, and top vs low performing film benchmarks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.ui import init_page, render_page_header, render_kpi_card, render_synthetic_data_banner
from src.config import COLORS
from src.analytics import load_enriched_dataset, get_luxury_chart_layout, create_release_decay_chart

# Initialize page
init_page("Movie Intelligence", icon="🎞️")

render_page_header(
    title="Movie Intelligence & Lifecycle Analytics",
    subtitle="Title-level box office analytics, audience rating elasticity, and release decay patterns",
    badge_text="FILM PERFORMANCE"
)

render_synthetic_data_banner()

@st.cache_data(show_spinner="Analyzing film portfolio...")
def get_data():
    return load_enriched_dataset()

df_full = get_data()

# ----------------- SIDEBAR FILTERS -----------------
with st.sidebar:
    st.markdown("### 🎛️ Filter Catalogue")
    all_genres = sorted(df_full["genre"].unique())
    selected_genres = st.multiselect("Genre Filter", options=all_genres, default=all_genres)

    all_theatres = sorted(df_full["theatre_name"].unique())
    selected_theatres = st.multiselect("Theatre Filter", options=all_theatres, default=all_theatres)

filtered_df = df_full[
    (df_full["genre"].isin(selected_genres)) &
    (df_full["theatre_name"].isin(selected_theatres))
]

if filtered_df.empty:
    st.warning("No screenings found matching the selected filters.")
    st.stop()

# ----------------- KPI SUMMARY -----------------
movie_agg = filtered_df.groupby("movie_title").agg(
    genre=("genre", "first"),
    total_screenings=("booking_id", "count"),
    total_tickets=("tickets_sold", "sum"),
    avg_tickets=("tickets_sold", "mean"),
    avg_occupancy=("actual_occupancy", "mean"),
    total_revenue=("actual_revenue", "sum"),
    avg_rating=("movie_rating", "mean"),
    max_days=("days_since_release", "max")
).reset_index()

top_grossing = movie_agg.sort_values("total_revenue", ascending=False).iloc[0]
top_occupancy = movie_agg.sort_values("avg_occupancy", ascending=False).iloc[0]
lowest_occupancy = movie_agg.sort_values("avg_occupancy", ascending=True).iloc[0]

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        render_kpi_card(
            label="Titles In Catalogue",
            value=f"{len(movie_agg)}",
            subtext=f"{filtered_df['booking_id'].count():,} Total Screenings"
        ),
        unsafe_allow_html=True
    )
with c2:
    st.markdown(
        render_kpi_card(
            label="Top Grossing Title",
            value=top_grossing["movie_title"].split(":")[0],
            subtext=f"${top_grossing['total_revenue']:,.0f} Gross Revenue",
            is_gold=True
        ),
        unsafe_allow_html=True
    )
with c3:
    st.markdown(
        render_kpi_card(
            label="Highest Fill Rate",
            value=f"{top_occupancy['avg_occupancy']:.1f}%",
            subtext=f"{top_occupancy['movie_title']} ({top_occupancy['genre']})"
        ),
        unsafe_allow_html=True
    )
with c4:
    st.markdown(
        render_kpi_card(
            label="Lowest Fill Rate",
            value=f"{lowest_occupancy['avg_occupancy']:.1f}%",
            subtext=f"{lowest_occupancy['movie_title']} ({lowest_occupancy['genre']})"
        ),
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ----------------- CHARTS ROW 1: TICKET SALES & POPULARITY SCATTER -----------------
col_bar, col_scatter = st.columns([1.2, 1.0])

with col_bar:
    sorted_movies = movie_agg.sort_values("total_tickets", ascending=True)
    fig_bar = go.Figure(go.Bar(
        y=sorted_movies["movie_title"],
        x=sorted_movies["total_tickets"],
        orientation="h",
        marker=dict(
            color=sorted_movies["avg_occupancy"],
            colorscale=[[0, "#E5DFD4"], [0.5, "#42675C"], [1.0, "#172321"]],
            colorbar=dict(title="Avg Occ %", ticksuffix="%")
        ),
        text=sorted_movies["total_tickets"].apply(lambda x: f" {x:,}"),
        textposition="outside",
        textfont=dict(color="#172321", size=10, family="Plus Jakarta Sans, sans-serif")
    ))
    bar_layout = get_luxury_chart_layout()
    bar_layout["xaxis"]["range"] = [0, sorted_movies["total_tickets"].max() * 1.18]
    fig_bar.update_layout(
        bar_layout,
        title="<b>Cumulative Ticket Sales by Title (Colored by Occupancy %)</b>",
        xaxis_title="Total Tickets Sold",
        yaxis_title="Movie Title",
        height=450
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with col_scatter:
    fig_scatter = px.scatter(
        movie_agg,
        x="avg_rating",
        y="avg_tickets",
        size="total_screenings",
        color="genre",
        hover_name="movie_title",
        labels={
            "avg_rating": "Audience Rating (IMDb Scale)",
            "avg_tickets": "Avg Tickets / Screening",
            "genre": "Genre"
        },
        title="<b>Movie Popularity vs Ticket Sales Elasticity</b>"
    )
    fig_scatter.update_layout(
        get_luxury_chart_layout(),
        height=450
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ----------------- CHARTS ROW 2: RELEASE AGE DECAY -----------------
st.plotly_chart(
    create_release_decay_chart(filtered_df),
    use_container_width=True
)

# ----------------- TOP VS LOW PERFORMERS TABLE -----------------
st.markdown("### 🏆 Performance Comparison: Top vs Low Performing Titles")

tab_top, tab_low = st.tabs(["Top-Performing Films (Highest Fill Rate)", "Underperforming Films (Optimization Needed)"])

with tab_top:
    top_5 = movie_agg.sort_values("avg_occupancy", ascending=False).head(5).copy()
    top_5["avg_revenue_per_show"] = top_5["total_revenue"] / top_5["total_screenings"]
    st.dataframe(
        top_5[[
            "movie_title", "genre", "avg_rating", "total_screenings",
            "total_tickets", "avg_occupancy", "total_revenue", "avg_revenue_per_show"
        ]].rename(columns={
            "movie_title": "Film Title",
            "genre": "Genre",
            "avg_rating": "Rating",
            "total_screenings": "Shows",
            "total_tickets": "Total Tickets",
            "avg_occupancy": "Avg Occupancy %",
            "total_revenue": "Total Revenue ($)",
            "avg_revenue_per_show": "Rev / Show ($)"
        }).style.format({
            "Rating": "{:.1f}",
            "Shows": "{:,}",
            "Total Tickets": "{:,}",
            "Avg Occupancy %": "{:.1f}%",
            "Total Revenue ($)": "${:,.2f}",
            "Rev / Show ($)": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

with tab_low:
    low_5 = movie_agg.sort_values("avg_occupancy", ascending=True).head(5).copy()
    low_5["avg_revenue_per_show"] = low_5["total_revenue"] / low_5["total_screenings"]
    st.dataframe(
        low_5[[
            "movie_title", "genre", "avg_rating", "total_screenings",
            "total_tickets", "avg_occupancy", "total_revenue", "avg_revenue_per_show"
        ]].rename(columns={
            "movie_title": "Film Title",
            "genre": "Genre",
            "avg_rating": "Rating",
            "total_screenings": "Shows",
            "total_tickets": "Total Tickets",
            "avg_occupancy": "Avg Occupancy %",
            "total_revenue": "Total Revenue ($)",
            "avg_revenue_per_show": "Rev / Show ($)"
        }).style.format({
            "Rating": "{:.1f}",
            "Shows": "{:,}",
            "Total Tickets": "{:,}",
            "Avg Occupancy %": "{:.1f}%",
            "Total Revenue ($)": "${:,.2f}",
            "Rev / Show ($)": "${:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )
