"""Feature definitions shared by training, the API and the dashboard."""
import pandas as pd

NUMERIC = [
    "dist_to_boundary_km", "dist_to_road_km", "dist_to_town_km", "dist_to_water_km", "dist_to_border_km",
    "dist_to_ranger_km", "species_records_10km", "target_species_records_10km",
    "elevation_m", "slope_deg", "ndvi", "forest_cover", "fragmentation_index", "reserve_pressure",
]
CATEGORICAL = ["land_cover", "season"]
FLAGS = ["has_tiger", "has_elephant", "has_rhino"]
CLASSES = ["Low", "Medium", "High"]

PRETTY = {
    "dist_to_boundary_km": "Distance to reserve edge",
    "dist_to_road_km": "Distance to road",
    "dist_to_town_km": "Distance to town",
    "dist_to_water_km": "Distance to river / lake",
    "dist_to_border_km": "Distance to international border",
    "dist_to_ranger_km": "Distance to ranger post",
    "species_records_10km": "Wildlife sightings nearby",
    "target_species_records_10km": "Target-species sightings nearby",
    "elevation_m": "Elevation",
    "slope_deg": "Terrain slope",
    "ndvi": "Vegetation greenness (NDVI)",
    "forest_cover": "Forest cover",
    "fragmentation_index": "Habitat fragmentation",
    "reserve_pressure": "Reserve's past poaching pressure",
    "land_cover": "Land cover", "season": "Season",
    "has_tiger": "Tiger reserve", "has_elephant": "Elephant habitat", "has_rhino": "Rhino habitat",
}

REAL_FEATURES = {"dist_to_boundary_km", "dist_to_road_km", "dist_to_town_km", "dist_to_water_km",
                 "dist_to_border_km", "species_records_10km", "target_species_records_10km"}


def add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["has_tiger"] = df["species"].str.contains("T").astype(int)
    df["has_elephant"] = df["species"].str.contains("E").astype(int)
    df["has_rhino"] = df["species"].str.contains("R").astype(int)
    return df


def X_of(df: pd.DataFrame) -> pd.DataFrame:
    df = add_flags(df) if "has_tiger" not in df else df
    return df[NUMERIC + CATEGORICAL + FLAGS]


def base_feature(transformed_name: str) -> str:
    """Map a one-hot column name (e.g. 'cat__land_cover_scrub') back to its source feature."""
    n = transformed_name.split("__", 1)[-1]
    for c in CATEGORICAL:
        if n.startswith(c + "_"):
            return c
    return n
