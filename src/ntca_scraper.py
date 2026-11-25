"""OPTIONAL - pull India's official tiger mortality records (NTCA) to replace the simulated
`reserve_pressure` feature and to validate the proxy labels.

Source: https://projecttiger.ntca.gov.in/tiger-mortality  (2012-2018 event table)
        https://tigernet.nic.in                            (later years)

The NTCA site could not be reached from the build machine, so this script was NOT tested against
the live page. It tries pandas.read_html on the page; if the layout differs, open the page in a
browser, save it as HTML, and pass the file:  python -m src.ntca_scraper saved_page.html

Output: data/raw/ntca_tiger_mortality.csv and data/processed/reserve_poaching_rate.csv
        (poaching deaths / all deaths per tiger reserve). Then in build_dataset.py replace the
        simulated `pressure` series with this rate, joined on reserve name.
"""
import sys
import pandas as pd
from .config import RAW, PROC

URL = "https://projecttiger.ntca.gov.in/tiger-mortality"


def main(src=URL):
    tables = pd.read_html(src)
    print(f"found {len(tables)} tables")
    df = max(tables, key=len)  # the event table is the longest one
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    df.to_csv(RAW / "ntca_tiger_mortality.csv", index=False)
    print(df.head())

    reserve_col = next((c for c in df.columns if "reserve" in c), None)
    cause_col = next((c for c in df.columns if "cause" in c or "status" in c), None)
    if reserve_col and cause_col:
        df["is_poaching"] = df[cause_col].astype(str).str.contains("poach|seiz", case=False)
        rate = df.groupby(reserve_col).agg(deaths=("is_poaching", "size"), poaching=("is_poaching", "sum"))
        rate["poaching_rate"] = (rate.poaching / rate.deaths).round(3)
        rate.to_csv(PROC / "reserve_poaching_rate.csv")
        print(rate.sort_values("poaching_rate", ascending=False).head(15))
    else:
        print("Could not find reserve / cause columns; inspect data/raw/ntca_tiger_mortality.csv by hand.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else URL)
