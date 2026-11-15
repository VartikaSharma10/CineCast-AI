"""
CineCast AI - Prediction Engine
Loads serialized preprocessing pipeline and champion ML model to generate
single-show and batch demand forecasts with physical boundary enforcement.
"""

import json
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from src.config import (
    BEST_MODEL_PATH,
    PIPELINE_PATH,
    MODEL_METADATA_PATH,
    HOLIDAYS_2025_2026
)

# In-memory cache for fast repeated predictions
_CACHED_MODEL = None
_CACHED_PIPELINE = None
_CACHED_METADATA = None


def load_artifacts():
    """
    Loads model, preprocessing pipeline, and metadata with in-memory caching.
    """
    global _CACHED_MODEL, _CACHED_PIPELINE, _CACHED_METADATA

    if _CACHED_MODEL is None or _CACHED_PIPELINE is None:
        if not BEST_MODEL_PATH.exists() or not PIPELINE_PATH.exists():
            raise FileNotFoundError(
                f"Model or pipeline artifacts not found at {BEST_MODEL_PATH}. "
                "Please run `python -m src.train_model` first."
            )
        _CACHED_MODEL = joblib.load(BEST_MODEL_PATH)
        _CACHED_PIPELINE = joblib.load(PIPELINE_PATH)

    if _CACHED_METADATA is None and MODEL_METADATA_PATH.exists():
        with open(MODEL_METADATA_PATH, "r") as f:
            _CACHED_METADATA = json.load(f)

    return _CACHED_MODEL, _CACHED_PIPELINE, _CACHED_METADATA


def categorize_demand(occupancy_pct: float) -> dict:
    """
    Categorizes demand based on projected occupancy percentage with high contrast accessible colors.
    """
    if occupancy_pct >= 85.0:
        return {
            "tier": "Very High",
            "badge_color": "#8C6B2B", # Rich Bronze Gold
            "text_color": "#FFFFFF",
            "summary": "Auditorium at near-peak capacity. High risk of sellout."
        }
    elif occupancy_pct >= 65.0:
        return {
            "tier": "High",
            "badge_color": "#245A4B", # Deep Forest Sage
            "text_color": "#FFFFFF",
            "summary": "Healthy demand above cinema operational break-even."
        }
    elif occupancy_pct >= 40.0:
        return {
            "tier": "Moderate",
            "badge_color": "#9E641B", # Deep Amber
            "text_color": "#FFFFFF",
            "summary": "Standard weekday or mid-run baseline demand."
        }
    else:
        return {
            "tier": "Low",
            "badge_color": "#8A1F1F", # Deep Crimson
            "text_color": "#FFFFFF",
            "summary": "Under-utilized screening capacity. Action recommended."
        }


def predict_single_show(show_params: dict) -> dict:
    """
    Generates demand prediction for a single upcoming movie screening.

    Parameters:
        show_params (dict):
            - movie_title (str)
            - genre (str)
            - theatre_name (str)
            - show_date (str 'YYYY-MM-DD' or datetime.date)
            - show_time (str 'HH:MM')
            - ticket_price (float)
            - theatre_capacity (int)
            - advance_bookings (int)
            - movie_rating (float)
            - days_since_release (int)
            - is_holiday (int, optional)

    Returns:
        dict:
            - predicted_tickets: int
            - occupancy_pct: float
            - expected_revenue: float
            - remaining_seats: int
            - demand_tier: str
            - demand_badge: dict
            - confidence_lower: int
            - confidence_upper: int
            - quick_recommendation: str
    """
    model, pipeline, metadata = load_artifacts()

    # Normalize inputs
    show_date_str = str(show_params.get("show_date"))[:10]
    show_time_str = str(show_params.get("show_time"))
    theatre_cap = int(show_params.get("theatre_capacity", 200))
    adv_bookings = int(show_params.get("advance_bookings", 0))
    ticket_price = float(show_params.get("ticket_price", 18.0))
    rating = float(show_params.get("movie_rating", 7.5))
    days_released = int(show_params.get("days_since_release", 7))

    # Derive calendar factors
    dt = datetime.strptime(show_date_str, "%Y-%m-%d")
    is_weekend = 1 if dt.weekday() in [4, 5, 6] else 0
    is_holiday = show_params.get("is_holiday")
    if is_holiday is None:
        is_holiday = 1 if show_date_str in HOLIDAYS_2025_2026 else 0
    else:
        is_holiday = int(is_holiday)

    # Build single-row DataFrame for pipeline
    input_df = pd.DataFrame([{
        "booking_id": "PRED-001",
        "movie_title": show_params.get("movie_title", "Custom Movie"),
        "genre": show_params.get("genre", "Action"),
        "theatre_name": show_params.get("theatre_name", "Grand Regal Cinema - Screen 1 (IMAX)"),
        "show_date": show_date_str,
        "show_time": show_time_str,
        "ticket_price": ticket_price,
        "theatre_capacity": theatre_cap,
        "advance_bookings": adv_bookings,
        "movie_rating": rating,
        "days_since_release": days_released,
        "is_weekend": is_weekend,
        "is_holiday": is_holiday,
    }])

    # Transform through pipeline
    X_trans = pipeline.transform(input_df)

    # Raw model prediction
    raw_pred = float(model.predict(X_trans)[0])

    # STRICT PHYSICAL CONSTRAINTS:
    # 1. At least advance bookings
    # 2. At most auditorium capacity
    # 3. Non-negative
    constrained_pred = int(round(np.clip(raw_pred, adv_bookings, theatre_cap)))
    constrained_pred = max(0, min(theatre_cap, constrained_pred))

    occupancy_pct = round((constrained_pred / theatre_cap) * 100, 1)
    expected_revenue = round(constrained_pred * ticket_price, 2)
    remaining_seats = max(0, theatre_cap - constrained_pred)

    demand_info = categorize_demand(occupancy_pct)

    # Confidence interval using model RMSE
    rmse = metadata["champion_metrics"]["RMSE"] if metadata else 20.0
    conf_lower = int(max(adv_bookings, round(constrained_pred - 0.75 * rmse)))
    conf_upper = int(min(theatre_cap, round(constrained_pred + 0.75 * rmse)))

    # Quick single-show operational recommendation
    if occupancy_pct >= 85:
        rec = f"High demand alert! Consider adding a supplementary late screening or upgrading to an IMAX auditorium."
    elif occupancy_pct >= 65:
        rec = f"Optimal occupancy. Maintain current scheduling; standard concession bundling will maximize yield."
    elif occupancy_pct >= 40:
        rec = f"Moderate traction. Consider early-bird concession coupons or loyalty app push notifications."
    else:
        rec = f"Low occupancy risk. Recommend evaluating ticket pricing discount ($2-$3 reduction) or rescheduling."

    return {
        "predicted_tickets": constrained_pred,
        "occupancy_pct": occupancy_pct,
        "expected_revenue": expected_revenue,
        "remaining_seats": remaining_seats,
        "demand_tier": demand_info["tier"],
        "demand_badge": demand_info,
        "confidence_lower": conf_lower,
        "confidence_upper": conf_upper,
        "quick_recommendation": rec,
        "input_summary": {
            "theatre_capacity": theatre_cap,
            "advance_bookings": adv_bookings,
            "ticket_price": ticket_price,
            "is_weekend": bool(is_weekend),
            "is_holiday": bool(is_holiday)
        }
    }


def batch_predict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Runs model inference across a full DataFrame of screenings,
    returning a copy enriched with predicted_tickets, predicted_revenue,
    predicted_occupancy, and demand_category.
    """
    model, pipeline, _ = load_artifacts()

    out_df = df.copy()
    X_trans = pipeline.transform(out_df)
    raw_preds = model.predict(X_trans)

    # Apply physical constraints row-by-row
    advances = out_df["advance_bookings"].values
    capacities = out_df["theatre_capacity"].values

    constrained_preds = np.clip(raw_preds, advances, capacities)
    constrained_preds = np.round(constrained_preds).astype(int)

    out_df["predicted_tickets"] = constrained_preds
    out_df["predicted_occupancy"] = np.round((out_df["predicted_tickets"] / out_df["theatre_capacity"]) * 100, 1)
    out_df["predicted_revenue"] = np.round(out_df["predicted_tickets"] * out_df["ticket_price"], 2)
    out_df["demand_category"] = out_df["predicted_occupancy"].apply(lambda occ: categorize_demand(occ)["tier"])

    return out_df


if __name__ == "__main__":
    sample_input = {
        "movie_title": "Dune: Prophecy",
        "genre": "Sci-Fi",
        "theatre_name": "Grand Regal Cinema - Screen 1 (IMAX)",
        "show_date": "2026-03-20",
        "show_time": "20:30",
        "ticket_price": 25.5,
        "theatre_capacity": 450,
        "advance_bookings": 240,
        "movie_rating": 8.7,
        "days_since_release": 3,
        "is_holiday": 0
    }
    result = predict_single_show(sample_input)
    print("Sample Single Show Prediction:")
    for k, v in result.items():
        print(f"  {k}: {v}")
