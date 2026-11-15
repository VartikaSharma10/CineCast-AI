"""
CineCast AI - Configuration and Design System Constants
Centralized paths, theme palette, and domain entities.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

RAW_DATA_PATH = DATA_DIR / "cinema_bookings.csv"
CLEAN_DATA_PATH = DATA_DIR / "cinema_bookings_clean.csv"
BEST_MODEL_PATH = MODELS_DIR / "best_model.pkl"
PIPELINE_PATH = MODELS_DIR / "preprocessing_pipeline.pkl"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"

# Visual Theme & Colors (Luxury Cinema Meets Enterprise Analytics - High Contrast)
COLORS = {
    "deep_forest": "#172321",    # Primary brand & dark accents
    "warm_ivory": "#F5F3ED",     # Background workspace
    "champagne_gold": "#8C6B2B", # Rich high-contrast bronze gold
    "champagne_light": "#DFCDB0",# Bright champagne for dark backgrounds
    "sage_green": "#2E584B",     # High-contrast readable sage/forest green
    "charcoal": "#172321",       # High-contrast typography & text
    "white": "#FFFFFF",          # Card background
    "card_border": "#DDD6C7",    # Soft subtle border
    "hover_bg": "#EDE8DE",       # Subtle hover
    "danger_red": "#8A1F1F",     # Negative / low demand badge (high contrast)
    "warning_amber": "#9E641B",  # Moderate demand badge (high contrast)
    "muted_gray": "#384A45"      # High-contrast secondary text
}

# Cinema Business Domain Entities
MOVIES = [
    {"title": "Dune: Prophecy", "genre": "Sci-Fi", "base_rating": 8.7, "runtime": 155},
    {"title": "Oppenheimer: The Dawn", "genre": "Drama", "base_rating": 9.1, "runtime": 180},
    {"title": "Cyberpunk: Neon City", "genre": "Sci-Fi", "base_rating": 8.2, "runtime": 138},
    {"title": "The Grand Royale", "genre": "Drama", "base_rating": 7.9, "runtime": 125},
    {"title": "Chronicles of Eldoria", "genre": "Action", "base_rating": 8.5, "runtime": 142},
    {"title": "Silent Echoes", "genre": "Horror", "base_rating": 7.2, "runtime": 104},
    {"title": "Velocity Rush", "genre": "Action", "base_rating": 7.8, "runtime": 118},
    {"title": "Midnight Whispers", "genre": "Thriller", "base_rating": 8.0, "runtime": 112},
    {"title": "The Cosmic Voyage", "genre": "Animation", "base_rating": 8.4, "runtime": 98},
    {"title": "Laughter Therapy", "genre": "Comedy", "base_rating": 7.4, "runtime": 105},
    {"title": "Shadows of Mystery", "genre": "Thriller", "base_rating": 7.6, "runtime": 120},
    {"title": "Family Ties", "genre": "Comedy", "base_rating": 7.1, "runtime": 110},
    {"title": "Autumn in Paris", "genre": "Romance", "base_rating": 7.7, "runtime": 115},
    {"title": "Into the Abyss", "genre": "Horror", "base_rating": 6.8, "runtime": 96},
]

GENRES = sorted(list({m["genre"] for m in MOVIES}))

THEATRES = [
    {"name": "Grand Regal Cinema - Screen 1 (IMAX)", "base_capacity": 450, "tier": "Premium IMAX"},
    {"name": "Grand Regal Cinema - Screen 2", "base_capacity": 280, "tier": "Standard"},
    {"name": "Apex Luxe Multiplex - Screen VIP", "base_capacity": 140, "tier": "Luxury VIP"},
    {"name": "Apex Luxe Multiplex - Screen 4", "base_capacity": 220, "tier": "Standard"},
    {"name": "Starlight Dome - Grand Auditorium", "base_capacity": 360, "tier": "Large Format"},
    {"name": "CineOasis Premiere - Hall Gold", "base_capacity": 180, "tier": "Gold Class"},
    {"name": "Velvet Screen Studios - Studio A", "base_capacity": 120, "tier": "Boutique"},
]

SHOWTIMES = [
    {"time": "10:00", "label": "Morning (10:00 AM)", "category": "Morning", "is_peak": 0},
    {"time": "13:30", "label": "Matinee (01:30 PM)", "category": "Matinee", "is_peak": 0},
    {"time": "17:00", "label": "Evening (05:00 PM)", "category": "Evening", "is_peak": 1},
    {"time": "20:30", "label": "Prime Night (08:30 PM)", "category": "Prime", "is_peak": 1},
    {"time": "23:15", "label": "Late Night (11:15 PM)", "category": "Late Night", "is_peak": 0},
]

HOLIDAYS_2025_2026 = {
    "2025-01-01": "New Year's Day",
    "2025-01-20": "Martin Luther King Jr. Day",
    "2025-02-14": "Valentine's Day",
    "2025-02-17": "Presidents' Day",
    "2025-05-26": "Memorial Day",
    "2025-07-04": "Independence Day",
    "2025-09-01": "Labor Day",
    "2025-10-31": "Halloween",
    "2025-11-27": "Thanksgiving Day",
    "2025-11-28": "Black Friday",
    "2025-12-24": "Christmas Eve",
    "2025-12-25": "Christmas Day",
    "2025-12-31": "New Year's Eve",
    "2026-01-01": "New Year's Day",
    "2026-02-14": "Valentine's Day",
    "2026-02-16": "Presidents' Day",
    "2026-05-25": "Memorial Day",
    "2026-07-04": "Independence Day",
}
