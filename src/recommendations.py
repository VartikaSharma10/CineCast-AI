"""
CineCast AI - Rule-Based Business Recommendation Engine
Generates deterministic, dynamic operational recommendations using actual
model predictions, occupancy metrics, pricing elasticity, and capacity benchmarks.
No LLM or fake AI - pure data-driven business decision support.
"""

import pandas as pd
import numpy as np


def generate_screening_recommendations(df: pd.DataFrame, max_items: int = 25) -> list:
    """
    Evaluates individual screenings against business rule thresholds.
    
    Rules:
    1. Capacity Overload / Extra Screening: Predicted occupancy >= 85%
    2. Underperforming Screening: Predicted occupancy < 40% and days_since_release <= 14
    3. Low Advance Booking Alert: Advance booking ratio < 18% with days_since_release <= 7
    4. Prime Dynamic Pricing Surge: Predicted occupancy >= 80%, is_weekend == 1, is_peak_showtime == 1
    5. Screen Downsizing Opportunity: Predicted tickets < 75 in auditorium with capacity >= 250
    """
    recommendations = []

    for _, row in df.iterrows():
        occ = row.get("predicted_occupancy", (row.get("predicted_tickets", 0) / max(1, row.get("theatre_capacity", 1))) * 100)
        tickets = int(row.get("predicted_tickets", row.get("tickets_sold", 0)))
        cap = int(row["theatre_capacity"])
        adv = int(row.get("advance_bookings", 0))
        price = float(row.get("ticket_price", 15.0))
        adv_ratio = (adv / cap) * 100 if cap > 0 else 0
        days_rel = int(row.get("days_since_release", 10))
        is_wknd = int(row.get("is_weekend", 0))
        is_peak = int(row.get("is_peak_showtime", 1 if row.get("show_time") in ["17:00", "20:30"] else 0))

        # Rule 1: High Occupancy -> Add Screening / Upsize
        if occ >= 85.0:
            overflow_potential = int(tickets * 0.20)
            potential_revenue_gain = round(overflow_potential * price, 2)
            recommendations.append({
                "booking_id": row.get("booking_id", "N/A"),
                "movie_title": row["movie_title"],
                "theatre_name": row["theatre_name"],
                "show_date": str(row["show_date"]),
                "show_time": str(row["show_time"]),
                "rule_type": "Capacity Surge / Additional Screening",
                "priority": "Urgent" if occ >= 92 else "High",
                "predicted_occupancy": f"{occ:.1f}%",
                "metric_trigger": f"Projected occupancy {occ:.1f}% exceeds 85% threshold ({tickets}/{cap} seats)",
                "action": "Schedule an auxiliary screening within 45 minutes or reallocate to a larger auditorium (e.g. IMAX 450 seats).",
                "projected_impact": f"Capture up to ~{overflow_potential} spillover patrons (+${potential_revenue_gain:,.0f} revenue)."
            })

        # Rule 2: Low Occupancy on Early Run -> Promotion & Concession Bundle
        elif occ < 40.0 and days_rel <= 14:
            discount_price = round(price * 0.82, 2)
            recommendations.append({
                "booking_id": row.get("booking_id", "N/A"),
                "movie_title": row["movie_title"],
                "theatre_name": row["theatre_name"],
                "show_date": str(row["show_date"]),
                "show_time": str(row["show_time"]),
                "rule_type": "Under-Utilized Slot / Promotional Push",
                "priority": "Medium",
                "predicted_occupancy": f"{occ:.1f}%",
                "metric_trigger": f"Early run (Day {days_rel}) projected occupancy only {occ:.1f}% (Under 40%)",
                "action": f"Launch flash digital discount (${price:.2f} -> ${discount_price:.2f}) or concession bundle (Popcorn + Drink combo).",
                "projected_impact": "Stimulate ~15-25% incremental walk-in attendance to recover fixed auditorium overhead."
            })

        # Rule 3: Low Advance Bookings Alert
        if adv_ratio < 16.0 and days_rel <= 7 and cap >= 200:
            recommendations.append({
                "booking_id": row.get("booking_id", "N/A"),
                "movie_title": row["movie_title"],
                "theatre_name": row["theatre_name"],
                "show_date": str(row["show_date"]),
                "show_time": str(row["show_time"]),
                "rule_type": "Pre-Booking Velocity Lag",
                "priority": "High",
                "predicted_occupancy": f"{occ:.1f}%",
                "metric_trigger": f"Advance bookings at only {adv_ratio:.1f}% of capacity ({adv}/{cap} seats reserved)",
                "action": "Push targeted email & mobile app push notification with seat selection highlight to cinema loyalty members.",
                "projected_impact": "Accelerate pre-show conversion and hedge against walk-in uncertainty."
            })

        # Rule 4: Prime Dynamic Pricing Surge
        if occ >= 80.0 and is_wknd == 1 and is_peak == 1:
            price_bump = 2.50
            rev_gain = round(tickets * price_bump, 2)
            recommendations.append({
                "booking_id": row.get("booking_id", "N/A"),
                "movie_title": row["movie_title"],
                "theatre_name": row["theatre_name"],
                "show_date": str(row["show_date"]),
                "show_time": str(row["show_time"]),
                "rule_type": "Dynamic Yield Optimization (Surge)",
                "priority": "Opportunity",
                "predicted_occupancy": f"{occ:.1f}%",
                "metric_trigger": f"Weekend peak screening with high demand index ({occ:.1f}% fill)",
                "action": f"Apply peak weekend surcharge (+${price_bump:.2f} to base ${price:.2f}).",
                "projected_impact": f"+${rev_gain:,.0f} direct gross profit gain without dampening attendance."
            })

        # Rule 5: Screen Downsizing / Relocation
        if tickets < 80 and cap >= 280 and occ < 35.0:
            recommendations.append({
                "booking_id": row.get("booking_id", "N/A"),
                "movie_title": row["movie_title"],
                "theatre_name": row["theatre_name"],
                "show_date": str(row["show_date"]),
                "show_time": str(row["show_time"]),
                "rule_type": "Auditorium Downsizing",
                "priority": "Medium",
                "predicted_occupancy": f"{occ:.1f}%",
                "metric_trigger": f"Only {tickets} patrons projected in a {cap}-seat auditorium ({occ:.1f}% fill)",
                "action": "Migrate screening to a boutique 120-seat studio; reserve large auditorium for high-traction titles.",
                "projected_impact": "Save HVAC and high-lumen projection utility costs while reallocating prime screen."
            })

    # Sort by priority
    priority_order = {"Urgent": 0, "High": 1, "Opportunity": 2, "Medium": 3}
    recommendations.sort(key=lambda r: priority_order.get(r["priority"], 99))

    return recommendations[:max_items]


def generate_macro_insights(df: pd.DataFrame) -> list:
    """
    Computes macro portfolio-level business recommendations.
    """
    insights = []

    # 1. Weekend vs Weekday Lift Analysis
    weekend_occ = df[df["is_weekend"] == 1]["actual_occupancy"].mean()
    weekday_occ = df[df["is_weekend"] == 0]["actual_occupancy"].mean()
    lift_pct = ((weekend_occ - weekday_occ) / weekday_occ) * 100 if weekday_occ > 0 else 0

    insights.append({
        "title": "Weekend Demand Surge & Capacity Allocation",
        "badge": "Macro Strategy",
        "color": "#172321",
        "description": (
            f"Weekend screenings achieve an average occupancy of {weekend_occ:.1f}%, representing a "
            f"+{lift_pct:.1f}% demand surge over weekday shows ({weekday_occ:.1f}%). "
            "Recommendation: Allocate maximum screen count to high-rating blockbusters starting Friday 4:00 PM "
            "through Sunday night, and run back-to-back Matinee slots."
        )
    })

    # 2. Genre Imbalance Analysis
    genre_occ = df.groupby("genre")["actual_occupancy"].mean()
    genre_share = df["genre"].value_counts(normalize=True) * 100
    top_genre = genre_occ.idxmax()
    top_genre_occ = genre_occ.max()
    top_genre_share = genre_share.get(top_genre, 0)

    insights.append({
        "title": f"Underserved Demand for {top_genre} Films",
        "badge": "Genre Optimization",
        "color": "#6F8980",
        "description": (
            f"{top_genre} films lead portfolio performance with an average occupancy of {top_genre_occ:.1f}%, "
            f"yet currently comprise only {top_genre_share:.1f}% of total scheduled screenings. "
            f"Recommendation: Increase {top_genre} show allocations by 15-20% during prime evening time slots."
        )
    })

    # 3. Off-Peak Showtime Optimization
    morning_occ = df[df["show_time"] == "10:00"]["actual_occupancy"].mean()
    prime_occ = df[df["show_time"] == "20:30"]["actual_occupancy"].mean()

    insights.append({
        "title": "Morning Slot Monetization (10:00 AM)",
        "badge": "Scheduling Optimization",
        "color": "#B69A68",
        "description": (
            f"Morning screenings lag at {morning_occ:.1f}% average occupancy compared to {prime_occ:.1f}% for Prime Night. "
            "Recommendation: Repurpose weekday morning slots for private corporate screenings, school educational matinees, "
            "or discounted 'Senior Morning Clubs' to generate auxiliary concession revenue."
        )
    })

    return insights
