"""Step 2 - build the 1 km grid, extract features, and construct proxy labels.

Output: data/processed/grid_cells.parquet  (one row per cell per season)

Feature provenance (be upfront about this in the report):
  REAL  - computed from public data in data/raw
            dist_to_road_km, dist_to_town_km, dist_to_water_km, dist_to_border_km  (Natural Earth 1:10m)
            dist_to_boundary_km                                                    (reserve footprint)
            species_records_10km, target_species_records_10km                      (GBIF occurrences)
  SIMULATED - spatially smooth placeholders until you plug in the real rasters
            elevation_m, slope_deg, ndvi, forest_cover, fragmentation_index,
            land_cover, dist_to_ranger_km, reserve_pressure
  Reserve footprints are approximate shapes generated around each reserve's
  centroid with its official area; replace with WDPA polygons when available.

Run:  python -m src.build_dataset
"""
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, Point
from shapely.ops import unary_union
from shapely import STRtree
from scipy.spatial import cKDTree

from .config import RAW, PROC, INDIA_CRS, CELL_M, SEED, ROOT

rng = np.random.default_rng(SEED)

LANDSCAPE_FOREST = {  # rough prior forest cover by landscape
    "Western Ghats": 0.80, "North-East": 0.72, "Eastern Ghats": 0.68, "Central India": 0.62,
    "Shivalik-Terai": 0.66, "Sundarbans": 0.55,
}
LAND_COVER = ["dense_forest", "open_forest", "grassland", "scrub", "wetland", "degraded"]


# ---------------------------------------------------------------- geometry helpers
def footprint(center_xy, area_km2, seed):
    """Irregular blob with the reserve's official area (approximate footprint)."""
    r = np.sqrt(area_km2 * 1e6 / np.pi)
    g = np.random.default_rng(seed)
    t = np.linspace(0, 2 * np.pi, 120, endpoint=False)
    rad = np.ones_like(t)
    for k in range(2, 6):
        rad += g.uniform(0.04, 0.16) / (k - 1) * np.sin(k * t + g.uniform(0, 2 * np.pi))
    stretch, rot = g.uniform(1.0, 1.6), g.uniform(0, np.pi)
    x, y = rad * np.cos(t) * stretch, rad * np.sin(t) / stretch
    xr, yr = x * np.cos(rot) - y * np.sin(rot), x * np.sin(rot) + y * np.cos(rot)
    poly = Polygon(np.c_[xr, yr]).buffer(0)
    scale = np.sqrt(np.pi / poly.area) * r / np.sqrt(np.pi)  # normalise to target area
    scale = np.sqrt(area_km2 * 1e6 / poly.area)
    from shapely import affinity
    poly = affinity.scale(poly, scale, scale, origin=(0, 0))
    return affinity.translate(poly, *center_xy)


def smooth_field(x, y, seed, n=6, scale_km=(6, 40)):
    """Cheap spatially-correlated noise in [-1, 1] (sum of random plane waves)."""
    g = np.random.default_rng(seed)
    z = np.zeros_like(x, dtype=float)
    for _ in range(n):
        lam = g.uniform(*scale_km) * 1000
        th = g.uniform(0, np.pi)
        z += np.sin((x * np.cos(th) + y * np.sin(th)) * 2 * np.pi / lam + g.uniform(0, 2 * np.pi))
    return z / np.sqrt(n / 2) / 2.2


def nearest_dist(points_xy, geoms):
    tree = STRtree(geoms)
    pts = [Point(p) for p in points_xy]
    idx = tree.nearest(pts)
    return np.array([pts[i].distance(geoms[j]) for i, j in enumerate(idx)])


# ---------------------------------------------------------------- main
def main():
    PROC.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(ROOT / "data" / "reserves_india.csv")
    res["reserve_id"] = np.arange(len(res))
    rg = gpd.GeoDataFrame(res, geometry=gpd.points_from_xy(res.lon, res.lat), crs=4326).to_crs(INDIA_CRS)

    countries = gpd.read_file(RAW / "ne_10m_admin_0_countries.geojson").to_crs(INDIA_CRS)
    india = countries[countries.ADM0_A3 == "IND"].geometry.buffer(0).union_all()
    region = india.buffer(400_000)
    neighbours = countries[(countries.ADM0_A3 != "IND") & countries.intersects(region)]
    nb = [g.buffer(0).intersection(region) for g in neighbours.geometry]
    border_geom = india.boundary.intersection(unary_union(nb).buffer(3000))

    # footprints, clipped to land
    polys = []
    for _, r in rg.iterrows():
        p = footprint((r.geometry.x, r.geometry.y), r.area_km2, SEED + r.reserve_id).intersection(india)
        polys.append(p)
    rg["footprint"] = polys

    # ---- grid
    cells = []
    for _, r in rg.iterrows():
        p = r.footprint
        minx, miny, maxx, maxy = p.bounds
        xs = np.arange(np.floor(minx / CELL_M) * CELL_M, maxx, CELL_M) + CELL_M / 2
        ys = np.arange(np.floor(miny / CELL_M) * CELL_M, maxy, CELL_M) + CELL_M / 2
        X, Y = np.meshgrid(xs, ys)
        pts = gpd.GeoSeries(gpd.points_from_xy(X.ravel(), Y.ravel()))
        inside = pts.within(p).values
        cx, cy = X.ravel()[inside], Y.ravel()[inside]
        bnd = p.boundary
        d_b = np.array([bnd.distance(Point(a, b)) for a, b in zip(cx, cy)])
        cells.append(pd.DataFrame({"reserve_id": r.reserve_id, "x": cx, "y": cy, "dist_to_boundary_km": d_b / 1000}))
    cdf = pd.concat(cells, ignore_index=True)
    cdf["cell_id"] = [f"C{i:06d}" for i in range(len(cdf))]
    print("cells:", len(cdf))
    xy = cdf[["x", "y"]].values

    # ---- REAL distance features (Natural Earth)
    bbox = gpd.GeoSeries([region], crs=INDIA_CRS)
    roads = gpd.read_file(RAW / "ne_10m_roads.geojson", bbox=bbox.to_crs(4326)).to_crs(INDIA_CRS)
    towns = gpd.read_file(RAW / "ne_10m_populated_places_simple.geojson").to_crs(INDIA_CRS)
    towns = towns[towns.intersects(region)]
    rivers = gpd.read_file(RAW / "ne_10m_rivers_lake_centerlines.geojson", bbox=bbox.to_crs(4326)).to_crs(INDIA_CRS)
    lakes = gpd.read_file(RAW / "ne_10m_lakes.geojson", bbox=bbox.to_crs(4326)).to_crs(INDIA_CRS)
    water = list(rivers.geometry) + list(lakes.geometry)

    cdf["dist_to_road_km"] = nearest_dist(xy, list(roads.geometry)) / 1000
    tk = cKDTree(np.c_[towns.geometry.x, towns.geometry.y])
    cdf["dist_to_town_km"] = tk.query(xy)[0] / 1000
    cdf["dist_to_water_km"] = nearest_dist(xy, water) / 1000
    bparts = [g for g in getattr(border_geom, "geoms", [border_geom]) if not g.is_empty]
    print("border parts", len(bparts), round(border_geom.length/1000), "km")
    cdf["dist_to_border_km"] = nearest_dist(xy, bparts) / 1000
    print("distances done")

    # ---- REAL GBIF densities
    gb = pd.read_csv(RAW / "gbif_india_occurrences.csv")
    gbg = gpd.GeoDataFrame(gb, geometry=gpd.points_from_xy(gb.lon, gb.lat), crs=4326).to_crs(INDIA_CRS)
    allt = cKDTree(np.c_[gbg.geometry.x, gbg.geometry.y])
    cdf["species_records_10km"] = np.array([len(v) for v in allt.query_ball_point(xy, 10_000)])
    rmeta = rg.set_index("reserve_id")
    tgt = np.zeros(len(cdf))
    for sp, code in [("tiger", "T"), ("elephant", "E"), ("rhino", "R")]:
        sub = gbg[gbg.species == sp]
        t = cKDTree(np.c_[sub.geometry.x, sub.geometry.y])
        has = cdf.reserve_id.map(lambda i: code in rmeta.loc[i, "species"]).values
        cnt = np.array([len(v) for v in t.query_ball_point(xy, 10_000)])
        tgt += cnt * has
    cdf["target_species_records_10km"] = tgt
    print("gbif done")

    # ---- SIMULATED environmental features
    land = cdf.reserve_id.map(rmeta.landscape)
    base_el = cdf.reserve_id.map(rmeta.elev_m).values.astype(float)
    n1, n2, n3, n4 = (smooth_field(cdf.x.values, cdf.y.values, SEED + k) for k in range(4))
    edge = np.exp(-cdf.dist_to_boundary_km.values / 4)
    cdf["elevation_m"] = np.clip(base_el * (1 + 0.35 * n1) + 120 * n1, 0, None).round(0)
    relief = np.where(base_el > 700, 14, np.where(base_el > 300, 8, 3))
    cdf["slope_deg"] = np.clip(relief * (1 + 0.8 * n2) + rng.normal(0, 1.2, len(cdf)), 0, 45).round(1)
    fc = land.map(LANDSCAPE_FOREST).values + 0.18 * n3 - 0.22 * edge - 0.08 * np.exp(-cdf.dist_to_road_km.values / 3)
    cdf["forest_cover"] = np.clip(fc + rng.normal(0, 0.04, len(cdf)), 0.02, 0.98).round(3)
    cdf["fragmentation_index"] = np.clip(0.55 * edge + 0.35 * (1 - cdf.forest_cover) + 0.15 * np.abs(n4)
                                         + 0.12 * np.exp(-cdf.dist_to_road_km.values / 4)
                                         + rng.normal(0, 0.04, len(cdf)), 0, 1).round(3)
    lc = np.select(
        [land.values == "Sundarbans", cdf.dist_to_water_km < 1.0, cdf.forest_cover > 0.72, cdf.forest_cover > 0.5,
         cdf.fragmentation_index > 0.62, n4 > 0.35],
        ["wetland", "wetland", "dense_forest", "open_forest", "degraded", "grassland"], default="scrub")
    cdf["land_cover"] = lc
    # rangers: 1 station per ~250 km2, placed near roads inside the reserve
    d_r = np.zeros(len(cdf))
    for rid, grp in cdf.groupby("reserve_id"):
        k = max(1, int(round(len(grp) / 250)))
        w = np.exp(-grp.dist_to_road_km.values / 5) + 0.05
        st = grp.sample(k, weights=w, random_state=SEED + rid)[["x", "y"]].values
        d_r[grp.index] = cKDTree(st).query(grp[["x", "y"]].values)[0] / 1000
    cdf["dist_to_ranger_km"] = d_r.round(2)
    # reserve-level historical pressure (placeholder for real NTCA / MIKE poaching rate per reserve)
    pressure = pd.Series(np.clip(rng.beta(2, 3.2, len(rg)), 0.03, 0.97), index=rg.reserve_id)
    cdf["reserve_pressure"] = cdf.reserve_id.map(pressure).round(3)

    # ---- seasons
    dry, wet = cdf.copy(), cdf.copy()
    dry["season"], wet["season"] = "dry", "wet"
    dry["ndvi"] = np.clip(0.25 + 0.45 * dry.forest_cover - 0.05 + 0.05 * n2, -0.1, 0.9).round(3)
    wet["ndvi"] = np.clip(0.35 + 0.5 * wet.forest_cover + 0.05 * n2, -0.1, 0.95).round(3)
    df = pd.concat([dry, wet], ignore_index=True)

    # ---- PROXY LABEL RULE (documented, see reports/label_rule.md)
    s = lambda v, k: np.exp(-v / k)
    is_dry = (df.season == "dry").astype(float)
    spp = np.log1p(df.target_species_records_10km) / np.log1p(df.target_species_records_10km.max())
    latent = (
        1.30 * s(df.dist_to_boundary_km, 3)               # edges are porous
        + 1.10 * s(df.dist_to_road_km, 6)                 # access by vehicle
        + 0.70 * s(df.dist_to_town_km, 20)                # markets / poacher base
        + 0.90 * spp                                      # where the target animals are
        + 0.90 * is_dry * s(df.dist_to_water_km, 2.5)     # dry-season waterhole concentration
        + 0.20 * (1 - is_dry) * s(df.dist_to_water_km, 2.5)
        + 0.60 * s(df.dist_to_border_km, 60)              # trafficking routes
        + 0.80 * df.fragmentation_index
        + 1.20 * df.reserve_pressure
        + 0.55 * (1 - s(df.dist_to_ranger_km, 5))         # far from patrol posts
        - 0.012 * df.slope_deg                            # steep = harder for poachers too
    )
    latent = latent + rng.normal(0, 0.28, len(df))
    z = (latent - latent.mean()) / latent.std()
    df["threat_score"] = (100 / (1 + np.exp(-1.6 * (z - 0.35)))).round(2)
    q_hi, q_med = np.quantile(df.threat_score, [0.85, 0.55])
    df["risk_label"] = np.select([df.threat_score >= q_hi, df.threat_score >= q_med], ["High", "Medium"], "Low")

    # lat/lon for display
    ll = gpd.GeoSeries(gpd.points_from_xy(df.x, df.y), crs=INDIA_CRS).to_crs(4326)
    df["lon"], df["lat"] = ll.x.round(5), ll.y.round(5)
    df = df.merge(res[["reserve_id", "name", "state", "species", "landscape"]], on="reserve_id")
    df.to_parquet(PROC / "grid_cells.parquet", index=False)
    print(df.risk_label.value_counts(normalize=True).round(3).to_dict(), "thresholds", round(q_med, 1), round(q_hi, 1))

    # footprints for the web map (lat/lon)
    fp = gpd.GeoDataFrame(res[["reserve_id", "name", "state", "species", "area_km2", "landscape"]],
                          geometry=list(rg.footprint), crs=INDIA_CRS).to_crs(4326)
    fp.to_file(PROC / "reserve_footprints.geojson", driver="GeoJSON")
    json.dump({"medium": float(q_med), "high": float(q_hi)}, open(PROC / "label_thresholds.json", "w"))


if __name__ == "__main__":
    main()
