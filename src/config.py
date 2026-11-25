"""Shared paths and constants for the Poaching Risk Predictor."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
WEB_DATA = ROOT / "web" / "data"

# Metric CRS for India (WGS 84 / UTM is per-zone; we use an India-wide Lambert Conformal Conic)
INDIA_CRS = "+proj=lcc +lat_1=12.472944 +lat_2=35.172806 +lat_0=24 +lon_0=80 +x_0=4000000 +y_0=4000000 +datum=WGS84 +units=m +no_defs"
CELL_M = 1000  # 1 km x 1 km grid cells
SEED = 42

NATURAL_EARTH = [
    "ne_10m_roads", "ne_10m_populated_places_simple", "ne_10m_rivers_lake_centerlines",
    "ne_10m_admin_0_countries", "ne_10m_lakes", "ne_10m_admin_1_states_provinces",
]
GBIF_SPECIES = {
    "tiger": "Panthera tigris",
    "elephant": "Elephas maximus",
    "rhino": "Rhinoceros unicornis",
    "leopard": "Panthera pardus",
}
