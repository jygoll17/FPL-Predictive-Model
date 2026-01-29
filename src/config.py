"""Configuration for FPL Data Collector."""

from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
ANALYSIS_DIR.mkdir(exist_ok=True)

# CSV file paths
PLAYERS_CSV = DATA_DIR / "players.csv"
CLUBS_CSV = DATA_DIR / "clubs.csv"
FIXTURE_HISTORY_CSV = DATA_DIR / "fixture_history.csv"
PLAYER_GAMEWEEK_CSV = DATA_DIR / "player_gameweek_stats.csv"
ID_MAPPING_JSON = DATA_DIR / "id_mappings.json"
METADATA_JSON = DATA_DIR / "metadata.json"

# FPL API endpoints
FPL_BASE_URL = "https://fantasy.premierleague.com/api"
FPL_BOOTSTRAP_URL = f"{FPL_BASE_URL}/bootstrap-static/"
FPL_FIXTURES_URL = f"{FPL_BASE_URL}/fixtures/"
FPL_ELEMENT_SUMMARY_URL = f"{FPL_BASE_URL}/element-summary"  # /{player_id}/

# Premier League website
PL_STATS_BASE_URL = "https://www.premierleague.com/stats/top/clubs"
PL_STATS_SEASON = "2024"  # Current season

# FPL-Data.co.uk
FPL_DATA_CSV_URL = "https://www.fpl-data.co.uk/statistics"

# HTTP settings
REQUEST_TIMEOUT = 30.0
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3

# User agent for web requests
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)

# Position mappings
POSITION_MAP = {
    1: "GKP",
    2: "DEF",
    3: "MID",
    4: "FWD",
}
