"""
CineCast AI - Analytics and Business Intelligence Module
Computes cinema operational KPIs, occupancy metrics, pricing elasticity,
and prepares Plotly visualizations styled with the luxury design system.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from src.config import CLEAN_DATA_PATH, COLORS
from src.prediction import batch_predict


def get_luxury_chart_layout():
    """
    Returns standard luxury theme layout dict for Plotly charts with high contrast text.
    """
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(245,243,237,0.5)",
        font=dict(family="Plus Jakarta Sans, -apple-system, sans-serif", color="#172321", size=12),
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(
            showgrid=True,
            gridcolor="#E2DDD2",
            linecolor="#B0A899",
            tickfont=dict(color="#172321", size=11),
            title=dict(font=dict(color="#172321", size=12, family="Plus Jakarta Sans")),
            zeroline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#E2DDD2",
            linecolor="#B0A899",
            tickfont=dict(color="#172321", size=11),
            title=dict(font=dict(color="#172321", size=12, family="Plus Jakarta Sans")),
            zeroline=False
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#172321")
        )
    )


def load_enriched_dataset() -> pd.DataFrame:
    """
    Loads clean screening data, computes revenue, occupancy, day of week,
    and runs batch predictions to enable actual vs predicted analytics.
    """
    df = pd.read_csv(CLEAN_DATA_PATH)
    
    # Date and calendar feature enrichment
    df["show_date_dt"] = pd.to_datetime(df["show_date"])
    df["day_of_week"] = df["show_date_dt"].dt.dayofweek # 0=Monday, 6=Sunday
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["day_name"] = df["day_of_week"].apply(lambda d: days[int(d)] if 0 <= int(d) < 7 else "Mon")
    df["month"] = df["show_date_dt"].dt.month

    # Financial & capacity metrics
    df["actual_revenue"] = df["tickets_sold"] * df["ticket_price"]
    df["actual_occupancy"] = (df["tickets_sold"] / df["theatre_capacity"]) * 100.0

    # Ensure batch predictions exist
    if "predicted_tickets" not in df.columns:
        df = batch_predict(df)
        df["predicted_revenue"] = df["predicted_tickets"] * df["ticket_price"]
        df["predicted_occupancy"] = (df["predicted_tickets"] / df["theatre_capacity"]) * 100.0

    df["occupancy_variance"] = df["actual_occupancy"] - df["predicted_occupancy"]
    df["revenue_variance"] = df["actual_revenue"] - df["predicted_revenue"]
    return df


def calculate_executive_kpis(df: pd.DataFrame) -> dict:
    """
    Calculates executive summary metrics.
    """
    total_historical_tickets = int(df["tickets_sold"].sum())
    total_historical_revenue = float(df["actual_revenue"].sum())
    predicted_upcoming_bookings = int(df["predicted_tickets"].sum())
    expected_ticket_revenue = float(df["predicted_revenue"].sum())
    average_theatre_occupancy = float(df["actual_occupancy"].mean())
    
    # Highest demand movie by average tickets per show
    movie_demand = df.groupby("movie_title")["tickets_sold"].mean()
    highest_demand_movie = movie_demand.idxmax() if not movie_demand.empty else "N/A"
    highest_demand_movie_avg = round(movie_demand.max(), 1) if not movie_demand.empty else 0

    # Peak demand showtime
    showtime_demand = df.groupby("show_time")["tickets_sold"].mean()
    peak_demand_showtime = showtime_demand.idxmax() if not showtime_demand.empty else "N/A"

    return {
        "total_historical_tickets": total_historical_tickets,
        "total_historical_revenue": total_historical_revenue,
        "predicted_upcoming_bookings": predicted_upcoming_bookings,
        "expected_ticket_revenue": expected_ticket_revenue,
        "average_theatre_occupancy": round(average_theatre_occupancy, 1),
        "highest_demand_movie": highest_demand_movie,
        "highest_demand_movie_avg": highest_demand_movie_avg,
        "peak_demand_showtime": peak_demand_showtime,
        "total_screenings": len(df)
    }


def create_actual_vs_predicted_trend_chart(df: pd.DataFrame) -> go.Figure:
    """
    Generates weekly time-series comparing actual vs predicted ticket sales.
    """
    df_sorted = df.copy()
    if "show_date_dt" not in df_sorted.columns:
        df_sorted["show_date_dt"] = pd.to_datetime(df_sorted["show_date"])
    df_sorted["week"] = df_sorted["show_date_dt"].dt.to_period("W").dt.start_time
    weekly = df_sorted.groupby("week")[["tickets_sold", "predicted_tickets"]].sum().reset_index()

    fig = go.Figure()
    
    # Actual Sales line
    fig.add_trace(go.Scatter(
        x=weekly["week"],
        y=weekly["tickets_sold"],
        name="Actual Tickets Sold",
        mode="lines+markers",
        line=dict(color=COLORS["deep_forest"], width=2.5),
        marker=dict(size=5, color=COLORS["deep_forest"])
    ))

    # Predicted Sales line (Rich high-contrast gold)
    fig.add_trace(go.Scatter(
        x=weekly["week"],
        y=weekly["predicted_tickets"],
        name="Model Predicted Demand",
        mode="lines",
        line=dict(color="#8C6B2B", width=2.5, dash="dash")
    ))

    fig.update_layout(
        get_luxury_chart_layout(),
        title="<b>Weekly Demand Trend: Actual vs Predicted Ticket Sales</b>",
        xaxis_title="Calendar Week",
        yaxis_title="Total Tickets Sold",
        hovermode="x unified"
    )
    return fig


def create_genre_demand_chart(df: pd.DataFrame) -> go.Figure:
    """
    Bar chart showing total tickets sold and average occupancy by genre.
    Text is placed outside the bars for guaranteed 100% legibility.
    """
    genre_stats = df.groupby("genre").agg(
        total_tickets=("tickets_sold", "sum"),
        avg_occupancy=("actual_occupancy", "mean"),
        screenings=("booking_id", "count")
    ).reset_index().sort_values("total_tickets", ascending=True)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=genre_stats["genre"],
        x=genre_stats["total_tickets"],
        orientation="h",
        name="Tickets Sold",
        marker=dict(
            color=COLORS["deep_forest"],
            line=dict(color="#8C6B2B", width=1)
        ),
        text=genre_stats["total_tickets"].apply(lambda v: f" {v:,}"),
        textposition="outside",
        textfont=dict(color="#172321", size=11, family="Plus Jakarta Sans, sans-serif")
    ))

    # Add space on x-axis for outside text
    max_x = genre_stats["total_tickets"].max() * 1.15

    layout = get_luxury_chart_layout()
    layout["xaxis"]["range"] = [0, max_x]

    fig.update_layout(
        layout,
        title="<b>Genre Demand Distribution (Total Tickets Sold)</b>",
        xaxis_title="Tickets Sold",
        yaxis_title="Genre"
    )
    return fig


def create_theatre_performance_chart(df: pd.DataFrame) -> go.Figure:
    """
    Grouped horizontal bar chart comparing average tickets sold and capacity by theatre.
    Uses grouped mode and outside labels to avoid blending.
    """
    theatre_stats = df.groupby("theatre_name").agg(
        avg_tickets=("tickets_sold", "mean"),
        avg_capacity=("theatre_capacity", "mean"),
        avg_occupancy=("actual_occupancy", "mean")
    ).reset_index().sort_values("avg_occupancy", ascending=True)

    fig = go.Figure()

    # Capacity bar
    fig.add_trace(go.Bar(
        y=theatre_stats["theatre_name"],
        x=theatre_stats["avg_capacity"],
        name="Seat Capacity",
        orientation="h",
        marker=dict(color="#C4BCAE", line=dict(color="#968C7D", width=1)),
        text=theatre_stats["avg_capacity"].apply(lambda v: f"{int(v)} seats"),
        textposition="outside",
        textfont=dict(color="#172321", size=10)
    ))

    # Average tickets sold bar
    fig.add_trace(go.Bar(
        y=theatre_stats["theatre_name"],
        x=theatre_stats["avg_tickets"],
        name="Avg Tickets Sold",
        orientation="h",
        marker=dict(color=COLORS["deep_forest"], line=dict(color="#8C6B2B", width=1)),
        text=theatre_stats["avg_tickets"].apply(lambda v: f"{int(v)} tix"),
        textposition="outside",
        textfont=dict(color="#172321", size=10)
    ))

    max_val = max(theatre_stats["avg_capacity"].max(), theatre_stats["avg_tickets"].max()) * 1.25

    layout = get_luxury_chart_layout()
    layout["xaxis"]["range"] = [0, max_val]

    fig.update_layout(
        layout,
        title="<b>Theatre Auditorium Utilization & Capacity Benchmarks</b>",
        barmode="group",
        xaxis_title="Seats / Tickets",
        yaxis_title="Theatre Auditorium"
    )
    return fig


def create_showtime_occupancy_heatmap(df: pd.DataFrame) -> go.Figure:
    """
    Heatmap of occupancy rate across Day of Week vs Showtime.
    Safely computes day_of_week if not present, and uses a high-contrast palette.
    """
    df_heat = df.copy()
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    # Guarantee day_of_week and day_name presence
    if "day_of_week" not in df_heat.columns or df_heat["day_of_week"].isna().any():
        df_heat["show_date_dt"] = pd.to_datetime(df_heat["show_date"])
        df_heat["day_of_week"] = df_heat["show_date_dt"].dt.dayofweek

    df_heat["day_name"] = df_heat["day_of_week"].apply(
        lambda d: days[int(d)] if pd.notna(d) and 0 <= int(d) < 7 else "Mon"
    )

    pivot = df_heat.pivot_table(
        index="show_time",
        columns="day_name",
        values="actual_occupancy",
        aggfunc="mean"
    )
    
    # Ensure all days are present in columns
    present_days = [d for d in days if d in pivot.columns]
    pivot = pivot[present_days]

    time_labels = {
        "10:00": "10:00 AM (Morning)",
        "13:30": "01:30 PM (Matinee)",
        "17:00": "05:00 PM (Evening)",
        "20:30": "08:30 PM (Prime)",
        "23:15": "11:15 PM (Late Night)"
    }
    pivot.index = [time_labels.get(t, t) for t in pivot.index]

    # Elegant high-contrast sequential colorscale (Ivory -> Sage -> Forest)
    colorscale = [
        [0.0, "#F7F5EE"],
        [0.25, "#D5E3DC"],
        [0.55, "#7A9E92"],
        [0.80, "#42675C"],
        [1.0, "#172321"]
    ]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values.round(1),
        x=pivot.columns.tolist(),
        y=pivot.index.tolist(),
        colorscale=colorscale,
        text=np.char.add(pivot.values.round(1).astype(str), "%"),
        texttemplate="%{text}",
        textfont=dict(size=12, family="Plus Jakarta Sans, sans-serif"),
        hoverongaps=False
    ))

    fig.update_layout(
        get_luxury_chart_layout(),
        title="<b>Showtime Occupancy Heatmap (% Capacity Utilized)</b>",
        xaxis_title="Day of Week",
        yaxis_title="Showtime"
    )
    return fig


def create_release_decay_chart(df: pd.DataFrame) -> go.Figure:
    """
    Scatter with trend line showing demand decay over days since release.
    """
    decay = df.groupby("days_since_release").agg(
        avg_tickets=("tickets_sold", "mean"),
        avg_occupancy=("actual_occupancy", "mean")
    ).reset_index().sort_values("days_since_release")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=decay["days_since_release"],
        y=decay["avg_occupancy"],
        mode="lines+markers",
        name="Average Occupancy %",
        line=dict(color="#8C6B2B", width=3),
        marker=dict(size=6, color=COLORS["deep_forest"])
    ))

    fig.update_layout(
        get_luxury_chart_layout(),
        title="<b>Box Office Lifecycle: Occupancy Decay vs Days Since Release</b>",
        xaxis_title="Days Since Theatrical Release",
        yaxis_title="Average Occupancy (%)"
    )
    return fig


def create_revenue_comparison_chart(df: pd.DataFrame) -> go.Figure:
    """
    Compares Actual vs Predicted revenue aggregated by movie title.
    """
    movie_rev = df.groupby("movie_title").agg(
        actual=("actual_revenue", "sum"),
        predicted=("predicted_revenue", "sum")
    ).reset_index().sort_values("actual", ascending=False)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=movie_rev["movie_title"],
        y=movie_rev["actual"],
        name="Actual Revenue ($)",
        marker=dict(color=COLORS["deep_forest"], line=dict(color="#172321", width=1))
    ))

    fig.add_trace(go.Bar(
        x=movie_rev["movie_title"],
        y=movie_rev["predicted"],
        name="Predicted Revenue ($)",
        marker=dict(color="#8C6B2B", line=dict(color="#6E4D1A", width=1))
    ))

    fig.update_layout(
        get_luxury_chart_layout(),
        title="<b>Theatrical Box Office Revenue: Actual vs Predicted by Title</b>",
        barmode="group",
        xaxis_title="Movie Title",
        yaxis_title="Total Revenue ($)",
        xaxis_tickangle=-30
    )
    return fig
