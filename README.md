# CineCast AI: Intelligent Cinema Ticket Demand Forecasting & Revenue Optimization Platform

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit 1.51](https://img.shields.io/badge/streamlit-1.51-red.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.8.0-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Placement-Drive Technical Assessment Capstone**  
> An end-to-end Machine Learning, Business Intelligence, and Dynamic Decision Support Platform built for cinema operators, multiplex chains, and theatrical distribution executives.

---

## 1. Project Overview & Business Problem

In theatrical cinema management, screening schedule optimization and auditorium capacity allocation directly govern profitability. Multiplexes face a perishable asset challenge: **an unsold seat when the projector turns on represents lost revenue that cannot be recovered**. 

Traditional cinema operations rely on manual heuristics and static booking estimates, resulting in:
- **Capacity Bottlenecks:** Blockbusters selling out with high spillover demand that could have justified auxiliary screens or larger IMAX auditoriums.
- **Under-Utilized Auditoriums:** Half-empty screens operating during off-peak slots with full HVAC, power, and projection costs.
- **Suboptimal Pricing:** Static ticket pricing that fails to capture premium weekend yield or stimulate soft weekday matinees.
- **Reactive Marketing:** Delayed realization that pre-sales are lagging, leaving insufficient time to run targeted promotional interventions.

**CineCast AI** solves this by establishing a predictive machine learning pipeline that models ticket sales using movie characteristics, release lifecycle stages, show timings, calendar patterns, and pricing elasticity.

---

## 2. Key Objectives & Business Value

1. **Demand Forecasting:** Predict accurate ticket sales for any upcoming screening before showtime.
2. **Revenue Optimization:** Quantify expected box office revenue and identify prime dynamic yield opportunities.
3. **Capacity Planning:** Provide seat fill projections and prevent auditorium overcapacity ($0 \le \hat{y} \le \text{Capacity}$).
4. **Lifecycle Analytics:** Quantify theatrical release decay curves from opening weekend through late runs.
5. **Dynamic Operational Directives:** Automatically trigger actionable interventions (e.g., adding screenings, dynamic discounting, pre-sales velocity warnings).

---

## 3. Technology Stack

- **Core Programming:** Python 3.11+
- **Web Application & UI/UX:** Streamlit (Multi-page architecture, Custom CSS styling injection)
- **Data Engineering:** Pandas, NumPy
- **Machine Learning & Pipeline:** Scikit-Learn (`GradientBoostingRegressor`, `RandomForestRegressor`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`)
- **Model Evaluation:** Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), $R^2$ Score
- **Interactive Visualizations:** Plotly Graph Objects, Plotly Express
- **Model Persistence:** Joblib
- **Storage:** CSV initial dataset storage with automated cleaning and deduplication

---

## 4. Dataset & Data Engineering

CineCast AI operates on a calibrated cinema screening dataset representing **6,500 historical screenings** across various multiplex auditoriums, release dates, and showtimes.

### Schema:
| Field | Type | Description |
|---|---|---|
| `booking_id` | String | Unique screening transaction identifier (`CC-100001`...) |
| `movie_title` | String | Title of the film (e.g., *Dune: Prophecy*, *Oppenheimer: The Dawn*) |
| `genre` | String | Film genre (Sci-Fi, Action, Drama, Comedy, Horror, etc.) |
| `theatre_name` | String | Specific auditorium (IMAX 450 seats, Luxury VIP 140 seats, etc.) |
| `show_date` | Date | Screening date (`YYYY-MM-DD`) |
| `show_time` | String | Scheduled showtime (`10:00`, `13:30`, `17:00`, `20:30`, `23:15`) |
| `ticket_price` | Float | Base ticket price per admission ($12.00 – $28.00) |
| `theatre_capacity` | Integer | Total seat capacity of the auditorium (120 – 450 seats) |
| `advance_bookings` | Integer | Confirmed pre-sales prior to showtime |
| `movie_rating` | Float | Critical/Audience rating benchmark (5.5 – 9.8) |
| `days_since_release` | Integer | Days since theatrical premiere (0 – 55 days) |
| `is_weekend` | Integer | Binary flag (1 for Friday, Saturday, Sunday) |
| `is_holiday` | Integer | Binary flag for calendar holidays |
| `tickets_sold` | Integer | **Target Variable:** Total admissions recorded ($0 \le y \le \text{Capacity}$) |

### Data Cleaning & Preprocessing Workflow (`src/preprocessing.py`):
1. **Deduplication:** Automatic removal of duplicate booking records.
2. **Missing Value Imputation:**
   - Missing `ticket_price` imputed via auditorium-specific medians.
   - Missing `movie_rating` imputed via title and genre-specific medians.
   - Missing `advance_bookings` imputed via capacity and median pre-booking ratios.
3. **Physical Bound Verification:** Strict programmatic assertion that $y \ge \text{advance\_bookings}$ and $y \le \text{theatre\_capacity}$.
4. **Chronological Ordering:** Sorted chronologically to mirror real production data pipelines.

---

## 5. Machine Learning Methodology

### Feature Engineering (`CinemaFeatureEngineer`):
- `day_of_week` & `month`: Calendar cyclical patterns.
- `show_hour`: Extracted integer hour of screening.
- `is_peak_showtime`: Binary flag for prime evening windows (17:00 – 21:30).
- `advance_booking_ratio`: $\frac{\text{advance\_bookings}}{\text{theatre\_capacity}}$ representing pre-show velocity.
- `price_per_rating`: Perceived value ratio ($\frac{\text{price}}{\text{rating}}$).
- `release_stage`: Categorical binning (Opening Weekend, Early Run, Mid Run, Late Run).
- `theatre_historical_occ`: Historical benchmark computed strictly on the training set to prevent leakage.

### Zero Temporal Leakage Validation:
To simulate real-world deployment, the dataset is partitioned using a **strict chronological train-test split** (80% training up to Dec 2025; 20% future testing from Dec 2025 to Mar 2026). Models are never trained on future dates.

### Model Comparison & Evaluation Results:
Three architectures were evaluated under identical chronological validation splits:

| Model | MAE (Tickets) | RMSE (Tickets) | $R^2$ Score | Status |
|---|:---:|:---:|:---:|:---:|
| **Baseline (Median Regressor)** | 48.22 | 65.72 | 0.1627 | Baseline Benchmark |
| **Random Forest Regressor** | 17.53 | 25.17 | 0.8771 | Candidate Architecture |
| **Gradient Boosting Regressor** | **14.94** | **20.62** | **0.9176** | **Champion Model Selected** |

The **Gradient Boosting Regressor** achieved superior generalization ($R^2 = 0.9176$, RMSE = 20.62), reducing prediction error by over **68%** relative to the operational baseline.

---

## 6. Application Architecture & Pages

CineCast AI is structured as a professional multi-page Streamlit enterprise suite:

```
cinemaforecast/
│
├── data/
│   ├── cinema_bookings.csv             # Raw generated dataset with simulated missing values
│   └── cinema_bookings_clean.csv       # Cleaned, validated, and deduplicated dataset
├── models/
│   ├── best_model.pkl                  # Serialized Champion Gradient Boosting Model
│   ├── preprocessing_pipeline.pkl      # Serialized Scikit-learn Feature Pipeline
│   └── model_metadata.json             # Model metrics, dates, and feature importances
├── src/
│   ├── __init__.py
│   ├── config.py                       # Theme palette, cinema constants, file paths
│   ├── data_generator.py               # Calibrated 6,500+ record dataset generator
│   ├── preprocessing.py                # Data cleaner and Feature Engineering Pipeline
│   ├── train_model.py                  # Chronological training and evaluation suite
│   ├── prediction.py                   # Single-show and batch inference engine
│   ├── recommendations.py              # Rule-based business decision support engine
│   ├── analytics.py                    # Aggregation algorithms and Plotly visualizations
│   └── ui.py                           # Layout injection, custom CSS, KPI cards
├── pages/
│   ├── 1_Executive_Overview.py         # C-Suite KPIs, weekly trends, genre distribution
│   ├── 2_Demand_Forecast.py            # Live interactive single-screening forecast studio
│   ├── 3_Movie_Intelligence.py         # Film box office rankings and release decay curves
│   ├── 4_Revenue_Analytics.py          # Yield analysis, actual vs predicted revenue variance
│   └── 5_Business_Recommendations.py   # Algorithmic operational queue with exportable actions
├── assets/
│   └── style.css                       # Luxury cinema design system styling
├── app.py                              # Application portal & model governance dashboard
├── requirements.txt                    # Python dependencies
├── .gitignore
└── README.md
```

### Module Breakdown:
1. **Executive Portal (`app.py`):** System health monitor, model lineage, chronological evaluation metrics table, and top feature importance visualization.
2. **Page 1: Executive Overview (`1_Executive_Overview.py`):** C-Suite KPI metrics, weekly actual vs predicted trends, genre distribution, auditorium utilization benchmarks, and upcoming schedule summary.
3. **Page 2: Demand Forecasting Studio (`2_Demand_Forecast.py`):** Interactive forecasting form with presets (Blockbuster Opening Night, Weekday Matinee, Family Matinee, Late Night Horror). Generates predicted sales, occupancy gauge, revenue, remaining seats, and immediate operational snippets.
4. **Page 3: Movie Intelligence (`3_Movie_Intelligence.py`):** Box office rankings, audience rating elasticity scatter, release-age decay curves, and top vs underperforming film tables.
5. **Page 4: Revenue & Occupancy Analytics (`4_Revenue_Analytics.py`):** Actual vs predicted revenue calibration, monthly box office trajectory, capacity saturation scatter with 100% capacity ceiling lines.
6. **Page 5: Smart Business Recommendations (`5_Business_Recommendations.py`):** Algorithmic operational directives (Auxiliary Screening, Promotional Push, Dynamic Surge, Downsizing) with priority filters and CSV action queue export.

---

## 7. Luxury Cinema Design System

CineCast AI strictly adheres to the requested **"Luxury Cinema meets Enterprise Analytics"** aesthetic:

- **Deep Forest Green (`#172321`):** Primary brand accent, navigation sidebar, and primary buttons.
- **Warm Ivory (`#F5F3ED`):** Clean, eye-pleasing workspace background.
- **Muted Champagne Gold (`#B69A68`):** Metrics highlights, active tabs, and demand badges.
- **Sage Green (`#6F8980`):** Positive indicators, operational status, and secondary metrics.
- **Charcoal (`#252B30`):** High-contrast typography.
- **White (`#FFFFFF`):** Elevated card surfaces with soft shadows and $1\text{px}$ subtle borders.

*Zero purple, neon, or generic Streamlit styling.*

---

## 8. Installation & Setup Instructions

### Prerequisites:
- Python 3.10, 3.11, or 3.12 installed on your system.

### Step 1: Clone or Navigate to Project Directory
```powershell
cd "c:\Users\Vartika Sharma\Desktop\cinemaforecast"
```

### Step 2: Install Required Dependencies
```powershell
pip install -r requirements.txt
```

### Step 3: (Optional) Re-generate Dataset and Retrain Model
The repository already includes the generated dataset and pre-trained champion model. To regenerate from scratch:
```powershell
# 1. Generate fresh calibrated screening dataset
python -m src.data_generator

# 2. Clean and preprocess data
python -m src.preprocessing

# 3. Train models, benchmark architectures, and serialize champion
python -m src.train_model
```

---

## 9. How to Run the Application

Execute the following command in your terminal:

```powershell
streamlit run app.py
```

The application will start immediately and launch in your default web browser at `http://localhost:8501`.

---

## 10. Verification & Quality Assurance

All components have been rigorously verified:
- [x] **Data Generation:** 6,500+ records generated with realistic genre, showtime, weekend, holiday, and release decay multipliers.
- [x] **Data Preprocessing:** Imputation of missing values and deduplication without data loss.
- [x] **Temporal Separation:** Chronological split preventing target leakage.
- [x] **Model Performance:** Gradient Boosting achieves $R^2 = 0.9176$ and $\text{RMSE} = 20.62$.
- [x] **Constraint Verification:** Ticket sales strictly bounded: $\max(\text{advance\_bookings}, \min(\text{capacity}, \hat{y}))$.
- [x] **UI/UX Consistency:** All 5 multi-page dashboards render with custom CSS and interactive Plotly visualizations.

---

## 11. Future Enhancements

1. **Weather API Integration:** Incorporate localized precipitation and temperature forecasts as real-time demand modifiers.
2. **Concessions & F&B Forecasting:** Extend model target from ticket admissions to predictive food & beverage concession spend per patron.
3. **Automated Cinema POS Webhooks:** Direct API integration with cinema Point of Sale (POS) systems (Vista, POSitive Cinema, RTS) for automated schedule sync.
