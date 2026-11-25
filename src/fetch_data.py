"""Step 1 - download the public datasets.

* Natural Earth 1:10m vectors (roads, towns, rivers, lakes, country borders) from GitHub.
* GBIF species occurrence records for tiger, Asian elephant, Indian rhino, leopard in India.

Run:  python -m src.fetch_data
"""
import json, time, urllib.request, urllib.parse
from .config import RAW, NATURAL_EARTH, GBIF_SPECIES

NE_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/{}.geojson"
GBIF_URL = "https://api.gbif.org/v1/occurrence/search"


def get(url):
    with urllib.request.urlopen(url, timeout=120) as r:
        return r.read()


def fetch_natural_earth():
    RAW.mkdir(parents=True, exist_ok=True)
    for name in NATURAL_EARTH:
        out = RAW / f"{name}.geojson"
        if out.exists():
            print("have", out.name); continue
        print("downloading", name)
        out.write_bytes(get(NE_URL.format(name)))


def fetch_gbif(max_per_species=3000):
    rows = []
    for key, sci in GBIF_SPECIES.items():
        offset, got = 0, 0
        while got < max_per_species:
            q = urllib.parse.urlencode({"scientificName": sci, "country": "IN", "hasCoordinate": "true",
                                        "hasGeospatialIssue": "false", "limit": 300, "offset": offset})
            data = json.loads(get(f"{GBIF_URL}?{q}"))
            for r in data["results"]:
                if "decimalLatitude" in r:
                    rows.append({"species": key, "lat": r["decimalLatitude"], "lon": r["decimalLongitude"],
                                 "year": r.get("year"), "basis": r.get("basisOfRecord"),
                                 "uncertainty_m": r.get("coordinateUncertaintyInMeters")})
            got += len(data["results"]); offset += 300
            if data.get("endOfRecords") or not data["results"]:
                break
            time.sleep(0.2)
        print(f"GBIF {key}: {got} records")
    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(RAW / "gbif_india_occurrences.csv", index=False)
    return df


if __name__ == "__main__":
    fetch_natural_earth()
    fetch_gbif()
