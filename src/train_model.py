"""
CineCast AI - Model Training, Evaluation, and Serialization Pipeline
Trains multiple regression models (Baseline, Random Forest, Gradient Boosting)
using chronological train-test splitting to prevent temporal leakage.
Calculates MAE, RMSE, and R2, saves the champion model and metadata.
"""

import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.config import (
    CLEAN_DATA_PATH,
    BEST_MODEL_PATH,
    PIPELINE_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR
)
from src.preprocessing import clean_raw_data, build_preprocessor


def train_and_evaluate_models():
    """
    Executes end-to-end model training, evaluation, and selection.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load cleaned dataset
    if not CLEAN_DATA_PATH.exists():
        print("[INFO] Clean dataset not found. Running data cleaner...")
        df = clean_raw_data()
    else:
        df = pd.read_csv(CLEAN_DATA_PATH)

    print(f"[INFO] Loaded {len(df)} records for training and validation.")

    # 2. Chronological Train-Test Split (80% train, 20% test)
    # Ensure chronological order
    df["datetime_sort"] = pd.to_datetime(df["show_date"] + " " + df["show_time"])
    df = df.sort_values("datetime_sort").reset_index(drop=True)
    df = df.drop(columns=["datetime_sort"])

    split_idx = int(len(df) * 0.80)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    train_dates = (train_df["show_date"].min(), train_df["show_date"].max())
    test_dates = (test_df["show_date"].min(), test_df["show_date"].max())

    print(f"[INFO] Chronological Split:")
    print(f"       Train period: {train_dates[0]} to {train_dates[1]} ({len(train_df)} rows)")
    print(f"       Test period:  {test_dates[0]} to {test_dates[1]} ({len(test_df)} rows)")

    y_train = train_df["tickets_sold"].values
    y_test = test_df["tickets_sold"].values

    # 3. Fit preprocessing pipeline
    pipeline = build_preprocessor()
    print("[INFO] Fitting feature engineering and preprocessing pipeline...")
    # Pass y_train to fit so transformer can compute historical benchmarks without leakage
    pipeline.named_steps["feature_engineer"].fit(train_df, y=y_train)
    
    # Transform training and testing feature sets
    X_train_trans = pipeline.fit_transform(train_df, y_train)
    X_test_trans = pipeline.transform(test_df)

    # Extract transformed feature names
    cat_encoder = pipeline.named_steps["encoder_scaler"].named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out())
    num_names = [
        "ticket_price", "theatre_capacity", "advance_bookings", "movie_rating",
        "days_since_release", "is_weekend", "is_holiday", "day_of_week",
        "month", "show_hour", "is_peak_showtime", "advance_booking_ratio",
        "price_per_rating", "theatre_historical_occ"
    ]
    feature_names = num_names + cat_names
    print(f"[INFO] Total engineered features: {len(feature_names)}")

    # 4. Model Candidates
    models = {
        "Baseline (Median)": DummyRegressor(strategy="median"),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=120,
            max_depth=14,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting Regressor": GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.07,
            max_depth=5,
            min_samples_split=4,
            min_samples_leaf=2,
            subsample=0.85,
            random_state=42
        )
    }

    results = {}
    fitted_models = {}

    print("\n" + "=" * 65)
    print(f"{'Model':<30} | {'MAE':<8} | {'RMSE':<8} | {'R2 Score':<8}")
    print("=" * 65)

    for name, model in models.items():
        model.fit(X_train_trans, y_train)
        preds = model.predict(X_test_trans)

        # Enforce physical constraints: advance_bookings <= preds <= theatre_capacity
        test_capacities = test_df["theatre_capacity"].values
        test_advances = test_df["advance_bookings"].values
        constrained_preds = np.clip(preds, test_advances, test_capacities)

        mae = float(mean_absolute_error(y_test, constrained_preds))
        rmse = float(root_mean_squared_error(y_test, constrained_preds))
        r2 = float(r2_score(y_test, constrained_preds))

        results[name] = {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "R2": round(r2, 4)
        }
        fitted_models[name] = model

        print(f"{name:<30} | {mae:<8.2f} | {rmse:<8.2f} | {r2:<8.4f}")

    print("=" * 65)

    # 5. Select Champion Model based on minimum RMSE on test set
    # Exclude baseline from selection
    ml_models = {k: v for k, v in results.items() if "Baseline" not in k}
    best_model_name = min(ml_models, key=lambda k: ml_models[k]["RMSE"])
    champion_model = fitted_models[best_model_name]
    print(f"\n[CHAMPION] Best Model Selected: {best_model_name} (RMSE: {results[best_model_name]['RMSE']}, R2: {results[best_model_name]['R2']})")

    # 6. Feature Importances
    importances_dict = {}
    if hasattr(champion_model, "feature_importances_"):
        raw_importances = champion_model.feature_importances_
        sorted_indices = np.argsort(raw_importances)[::-1]
        for idx in sorted_indices[:15]:
            importances_dict[feature_names[idx]] = round(float(raw_importances[idx]), 4)

    # 7. Persist Artifacts
    joblib.dump(champion_model, BEST_MODEL_PATH)
    joblib.dump(pipeline, PIPELINE_PATH)
    print(f"[OK] Saved model to {BEST_MODEL_PATH}")
    print(f"[OK] Saved pipeline to {PIPELINE_PATH}")

    metadata = {
        "best_model_name": best_model_name,
        "metrics": results,
        "champion_metrics": results[best_model_name],
        "feature_names": feature_names,
        "top_feature_importances": importances_dict,
        "train_date_range": train_dates,
        "test_date_range": test_dates,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "trained_at": pd.Timestamp.now().isoformat()
    }

    with open(MODEL_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[OK] Saved model metadata to {MODEL_METADATA_PATH}")

    return metadata


if __name__ == "__main__":
    train_and_evaluate_models()
