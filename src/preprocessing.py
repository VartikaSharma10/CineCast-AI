"""
CineCast AI - Preprocessing and Feature Engineering Pipeline
Handles data cleaning, missing value imputation, chronological ordering,
feature engineering, and Scikit-learn Pipeline construction.
Prevents target leakage and handles single-instance inference gracefully.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from src.config import RAW_DATA_PATH, CLEAN_DATA_PATH, HOLIDAYS_2025_2026


class CinemaFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to engineer cinema demand features from raw show metadata.
    Ensures zero target leakage and supports both bulk training and single prediction inference.
    """
    def __init__(self):
        self.theatre_benchmarks_ = {}

    def fit(self, X, y=None):
        # Compute baseline historical theatre fill benchmarks if target y or tickets_sold is present
        # strictly from training data to avoid leakage
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if y is not None:
                df["target_temp"] = y
                if "theatre_capacity" in df.columns:
                    df["occ_temp"] = df["target_temp"] / df["theatre_capacity"]
                    self.theatre_benchmarks_ = df.groupby("theatre_name")["occ_temp"].mean().to_dict()
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            df = pd.DataFrame(X)
        else:
            df = X.copy()

        # Parse date and time
        df["show_date_dt"] = pd.to_datetime(df["show_date"])
        df["day_of_week"] = df["show_date_dt"].dt.dayofweek
        df["month"] = df["show_date_dt"].dt.month
        df["day"] = df["show_date_dt"].dt.day

        # Weekend indicator (Friday=4, Saturday=5, Sunday=6)
        if "is_weekend" not in df.columns or df["is_weekend"].isna().any():
            df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [4, 5, 6] else 0)

        # Holiday indicator
        if "is_holiday" not in df.columns or df["is_holiday"].isna().any():
            df["is_holiday"] = df["show_date"].apply(lambda d: 1 if str(d)[:10] in HOLIDAYS_2025_2026 else 0)

        # Showtime hour & peak showtime indicator
        # show_time can be string "17:00" or similar
        def extract_hour(t_str):
            try:
                return int(str(t_str).split(":")[0])
            except Exception:
                return 17

        df["show_hour"] = df["show_time"].apply(extract_hour)
        # Peak hours: 17:00 (5 PM) to 21:30 (9:30 PM)
        df["is_peak_showtime"] = df["show_hour"].apply(lambda h: 1 if 17 <= h <= 21 else 0)

        # Advance booking ratio
        df["advance_booking_ratio"] = df["advance_bookings"] / df["theatre_capacity"].clip(lower=1)
        df["advance_booking_ratio"] = df["advance_booking_ratio"].clip(0.0, 1.0)

        # Price-to-rating index: measures perceived value
        df["price_per_rating"] = df["ticket_price"] / df["movie_rating"].clip(lower=1.0)

        # Release lifecycle category: 0=Opening (0-3d), 1=Early (4-10d), 2=Mid (11-24d), 3=Late (25+d)
        def categorize_release_age(days):
            if days <= 3:
                return "Opening_Weekend"
            elif days <= 10:
                return "Early_Run"
            elif days <= 24:
                return "Mid_Run"
            else:
                return "Late_Run"

        df["release_stage"] = df["days_since_release"].apply(categorize_release_age)

        # Benchmark theatre historical fill rate (defaults to 0.55 if unseen)
        df["theatre_historical_occ"] = df["theatre_name"].map(self.theatre_benchmarks_).fillna(0.55)

        # Select relevant transformed columns
        output_cols = [
            "movie_title",
            "genre",
            "theatre_name",
            "show_time",
            "ticket_price",
            "theatre_capacity",
            "advance_bookings",
            "movie_rating",
            "days_since_release",
            "is_weekend",
            "is_holiday",
            "day_of_week",
            "month",
            "show_hour",
            "is_peak_showtime",
            "advance_booking_ratio",
            "price_per_rating",
            "release_stage",
            "theatre_historical_occ"
        ]

        return df[output_cols]


def clean_raw_data(raw_csv_path=RAW_DATA_PATH, save_clean_csv=True) -> pd.DataFrame:
    """
    Cleans raw dataset:
    - Removes duplicate records
    - Imputes missing ticket_price by theatre median
    - Imputes missing movie_rating by movie or genre median
    - Imputes missing advance_bookings by theatre capacity benchmark
    - Validates target variable consistency
    """
    df = pd.read_csv(raw_csv_path)
    initial_count = len(df)

    # 1. Deduplication
    df = df.drop_duplicates(subset=["booking_id"]).reset_index(drop=True)
    dedup_count = len(df)

    # 2. Impute missing ticket_price with theatre-specific median
    theatre_price_median = df.groupby("theatre_name")["ticket_price"].transform("median")
    df["ticket_price"] = df["ticket_price"].fillna(theatre_price_median).fillna(df["ticket_price"].median())

    # 3. Impute missing movie_rating with title median, then genre median
    title_rating_median = df.groupby("movie_title")["movie_rating"].transform("median")
    genre_rating_median = df.groupby("genre")["movie_rating"].transform("median")
    df["movie_rating"] = df["movie_rating"].fillna(title_rating_median).fillna(genre_rating_median).fillna(7.5)

    # 4. Impute missing advance_bookings based on capacity & median advance ratio
    df["temp_adv_ratio"] = df["advance_bookings"] / df["theatre_capacity"]
    median_adv_ratio = df["temp_adv_ratio"].median()
    df["advance_bookings"] = df["advance_bookings"].fillna((df["theatre_capacity"] * median_adv_ratio).round())
    df = df.drop(columns=["temp_adv_ratio"])

    # 5. Physical bounds checks
    df["advance_bookings"] = df["advance_bookings"].clip(lower=0)
    df["advance_bookings"] = np.minimum(df["advance_bookings"], df["theatre_capacity"])
    
    if "tickets_sold" in df.columns:
        df["tickets_sold"] = df["tickets_sold"].clip(lower=0)
        # Ensure tickets_sold >= advance_bookings and <= capacity
        df["tickets_sold"] = np.maximum(df["tickets_sold"], df["advance_bookings"])
        df["tickets_sold"] = np.minimum(df["tickets_sold"], df["theatre_capacity"])
        df["tickets_sold"] = df["tickets_sold"].round().astype(int)

    # 6. Chronological sort
    df["datetime_sort"] = pd.to_datetime(df["show_date"] + " " + df["show_time"])
    df = df.sort_values("datetime_sort").reset_index(drop=True)
    df = df.drop(columns=["datetime_sort"])

    if save_clean_csv:
        CLEAN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CLEAN_DATA_PATH, index=False)
        print(f"[OK] Cleaned dataset saved to {CLEAN_DATA_PATH}")
        print(f"     Initial records: {initial_count} | Post-dedup: {dedup_count} | Final cleaned: {len(df)}")

    return df


def build_preprocessor():
    """
    Builds the ColumnTransformer for categorical and numerical features.
    """
    categorical_features = ["genre", "theatre_name", "show_time", "release_stage"]
    numerical_features = [
        "ticket_price",
        "theatre_capacity",
        "advance_bookings",
        "movie_rating",
        "days_since_release",
        "is_weekend",
        "is_holiday",
        "day_of_week",
        "month",
        "show_hour",
        "is_peak_showtime",
        "advance_booking_ratio",
        "price_per_rating",
        "theatre_historical_occ"
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
        ],
        remainder="drop"
    )

    full_pipeline = Pipeline([
        ("feature_engineer", CinemaFeatureEngineer()),
        ("encoder_scaler", preprocessor)
    ])

    return full_pipeline


if __name__ == "__main__":
    cleaned_df = clean_raw_data()
    print("Cleaned data sample:")
    print(cleaned_df.head(3))
