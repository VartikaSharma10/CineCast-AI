"""
CineCast AI - Synthetic Cinema Booking Dataset Generator
Generates realistic cinema screening data with genuine business dynamics:
- Weekend and holiday surges
- Opening week spikes and exponential release-age decay
- Showtime preference (Evening and Prime Night peaks)
- Price sensitivity and auditorium capacity constraints
- Realistic advance booking behavior
- Introduces controlled missing values and anomalies for preprocessing demonstration
"""

import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from src.config import MOVIES, THEATRES, SHOWTIMES, HOLIDAYS_2025_2026, RAW_DATA_PATH


def generate_cinema_dataset(num_records: int = 6500, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic historical screening dataset following realistic industry dynamics.
    
    Parameters:
        num_records (int): Target number of screening records (>= 5000)
        random_seed (int): Seed for reproducibility
        
    Returns:
        pd.DataFrame: Generated dataset saved to data/cinema_bookings.csv
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    start_date = datetime(2025, 1, 1)
    end_date = datetime(2026, 3, 15)
    total_days = (end_date - start_date).days

    # Genre popularity weightings
    genre_multipliers = {
        "Sci-Fi": 1.15,
        "Action": 1.20,
        "Drama": 0.95,
        "Comedy": 0.90,
        "Horror": 1.05,
        "Thriller": 1.00,
        "Animation": 1.10,
        "Romance": 0.85
    }

    # Showtime demand multipliers
    showtime_multipliers = {
        "10:00": 0.65,
        "13:30": 0.88,
        "17:00": 1.25,
        "20:30": 1.45,
        "23:15": 0.80
    }

    records = []

    for i in range(num_records):
        booking_id = f"CC-{100000 + i}"
        
        # Select random date within date range
        random_day_offset = random.randint(0, total_days)
        current_date = start_date + timedelta(days=random_day_offset)
        show_date_str = current_date.strftime("%Y-%m-%d")
        
        # Day of week and weekend/holiday checks
        day_of_week = current_date.weekday() # 0 = Monday, 6 = Sunday
        is_weekend = 1 if day_of_week in [4, 5, 6] else 0  # Friday, Saturday, Sunday
        is_holiday = 1 if show_date_str in HOLIDAYS_2025_2026 else 0
        
        # Choose Movie
        movie_info = random.choice(MOVIES)
        movie_title = movie_info["title"]
        genre = movie_info["genre"]
        base_rating = movie_info["base_rating"]
        
        # Ratings have slight screening-to-screening audience variance (+/- 0.3)
        movie_rating = round(float(np.clip(base_rating + np.random.normal(0, 0.2), 5.5, 9.8)), 1)
        
        # Days since release (simulate release lifecycle between 0 and 55 days)
        # Higher density in first 3 weeks
        if random.random() < 0.45:
            days_since_release = random.randint(0, 7)  # Opening week
        elif random.random() < 0.75:
            days_since_release = random.randint(8, 21) # Second-third week
        else:
            days_since_release = random.randint(22, 50) # Late run
            
        # Choose Theatre
        theatre_info = random.choice(THEATRES)
        theatre_name = theatre_info["name"]
        theatre_capacity = theatre_info["base_capacity"]
        theatre_tier = theatre_info["tier"]
        
        # Choose Showtime
        showtime_info = random.choice(SHOWTIMES)
        show_time = showtime_info["time"]
        
        # Ticket price: depends on theatre tier, peak time, and weekend
        base_price = 15.0
        if "IMAX" in theatre_tier:
            base_price = 24.0
        elif "VIP" in theatre_tier or "Gold" in theatre_tier:
            base_price = 28.0
        elif "Large Format" in theatre_tier:
            base_price = 19.5
        elif "Boutique" in theatre_tier:
            base_price = 21.0
        else:
            base_price = 14.5
            
        # Peak / weekend price surcharge
        price_adj = 0.0
        if is_weekend:
            price_adj += 2.50
        if showtime_info["is_peak"]:
            price_adj += 1.50
        ticket_price = round(base_price + price_adj + np.random.uniform(-1.0, 1.5), 2)
        
        # DEMAND DYNAMICS:
        # 1. Base demand factor (0.35 to 0.65)
        rating_factor = (movie_rating - 5.0) / 4.8 # normalized 0 to 1
        genre_factor = genre_multipliers.get(genre, 1.0)
        
        # 2. Release age decay curve
        # Strong opening (days 0-3), healthy first week, exponential drop off afterwards
        if days_since_release <= 3:
            release_decay = 1.35
        elif days_since_release <= 7:
            release_decay = 1.15
        elif days_since_release <= 14:
            release_decay = 0.95
        elif days_since_release <= 28:
            release_decay = 0.75
        else:
            release_decay = max(0.40, 0.70 - (days_since_release - 28) * 0.015)
            
        # 3. Calendar multipliers
        weekend_mult = 1.40 if is_weekend else 1.0
        if day_of_week == 5: # Saturday peak
            weekend_mult = 1.55
        holiday_mult = 1.45 if is_holiday else 1.0
        
        # 4. Showtime multiplier
        time_mult = showtime_multipliers.get(show_time, 1.0)
        # Late night horror boost
        if genre == "Horror" and show_time in ["20:30", "23:15"]:
            time_mult *= 1.25
        # Animation matinee boost on weekends
        if genre == "Animation" and is_weekend and show_time in ["10:00", "13:30"]:
            time_mult *= 1.40
            
        # 5. Price elasticity (higher price slightly reduces fill proportion)
        price_sensitivity = 1.0 - ((ticket_price - 14.0) / 70.0)
        price_sensitivity = np.clip(price_sensitivity, 0.80, 1.05)
        
        # Expected base fill rate (percentage of capacity)
        fill_rate = 0.42 * (0.6 + 0.4 * rating_factor) * genre_factor * release_decay * weekend_mult * holiday_mult * time_mult * price_sensitivity
        fill_rate += np.random.normal(0, 0.06) # realistic noise
        fill_rate = float(np.clip(fill_rate, 0.10, 0.98))
        
        # Advance bookings: proportion of final sales already booked in advance
        # Typically 30% to 75% of expected sales are booked in advance
        advance_ratio = float(np.clip(np.random.beta(4, 3) * (0.4 + 0.5 * rating_factor), 0.15, 0.85))
        expected_tickets = int(fill_rate * theatre_capacity)
        advance_bookings = int(expected_tickets * advance_ratio)
        advance_bookings = max(5, min(advance_bookings, theatre_capacity - 10))
        
        # Walk-in / on-the-day sales
        walk_in_demand = int(expected_tickets * (1 - advance_ratio) + np.random.normal(0, 8))
        walk_in_demand = max(0, walk_in_demand)
        
        tickets_sold = advance_bookings + walk_in_demand
        
        # STRICT PHYSICAL CONSTRAINTS:
        # Never exceed capacity
        tickets_sold = min(tickets_sold, theatre_capacity)
        # Never less than advance bookings
        tickets_sold = max(tickets_sold, advance_bookings)
        # Non-negative
        tickets_sold = max(0, tickets_sold)
        
        records.append({
            "booking_id": booking_id,
            "movie_title": movie_title,
            "genre": genre,
            "theatre_name": theatre_name,
            "show_date": show_date_str,
            "show_time": show_time,
            "ticket_price": ticket_price,
            "theatre_capacity": theatre_capacity,
            "advance_bookings": advance_bookings,
            "movie_rating": movie_rating,
            "days_since_release": days_since_release,
            "is_weekend": is_weekend,
            "is_holiday": is_holiday,
            "tickets_sold": tickets_sold
        })

    df = pd.DataFrame(records)
    
    # Sort chronologically by show_date and show_time
    df["datetime_sort"] = pd.to_datetime(df["show_date"] + " " + df["show_time"])
    df = df.sort_values("datetime_sort").reset_index(drop=True)
    df = df.drop(columns=["datetime_sort"])
    
    # Introduce controlled missing values (~1.5% in select columns) for preprocessing demonstration
    n_missing = int(len(df) * 0.015)
    missing_indices_price = np.random.choice(df.index, n_missing, replace=False)
    missing_indices_adv = np.random.choice(df.index, n_missing, replace=False)
    missing_indices_rate = np.random.choice(df.index, n_missing, replace=False)
    
    df.loc[missing_indices_price, "ticket_price"] = np.nan
    df.loc[missing_indices_adv, "advance_bookings"] = np.nan
    df.loc[missing_indices_rate, "movie_rating"] = np.nan
    
    # Add a few duplicated rows to demonstrate deduplication in pipeline
    duplicate_rows = df.sample(n=12, random_state=random_seed)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"[OK] Generated {len(df)} records saved to {RAW_DATA_PATH}")
    print(f"     Columns: {list(df.columns)}")
    print(f"     Missing ticket_price: {df['ticket_price'].isna().sum()}")
    print(f"     Missing advance_bookings: {df['advance_bookings'].isna().sum()}")
    print(f"     Missing movie_rating: {df['movie_rating'].isna().sum()}")
    return df


if __name__ == "__main__":
    generate_cinema_dataset(num_records=6500)
