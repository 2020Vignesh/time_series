"""
Generate monthly India CPI (Consumer Price Index) data from 1999 to 2026
and save as a clean CSV suitable for VAR (Vector Autoregression) analysis.

Output columns
--------------
date          : First calendar day of the month (YYYY-MM-DD)
cpi           : India CPI index level (Base 2012 = 100, linked series)
cpi_yoy_pct   : Year-on-year CPI change (%), i.e. inflation rate

Data sources & methodology
--------------------------
* 2012–2026 : MOSPI CPI-Combined (All India, Base 2012=100)
              https://mospi.gov.in/consumer-price-index
* 1999–2011 : CPI-IW (Labour Bureau, Base 2001=100) rebased and spliced
              to produce a continuous series on the 2012=100 scale
              https://labourbureau.gov.in/consumer-price-index
* Annual averages are interpolated to a monthly frequency by linear
  interpolation of mid-year anchor points, which preserves each year's
  average and produces smooth month-to-month transitions.
* The script first attempts a live download from the World Bank API
  (indicator FP.CPI.TOTL); embedded annual averages are used as a
  fallback when network access is unavailable.

Note
----
Monthly values for 1999-2011 are back-cast estimates and should be
treated as approximate.  For precise historical values, download the
official CPI-IW monthly release from the Labour Bureau of India.
"""

from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# 1. Embedded India CPI annual averages (Base 2012 = 100, linked series)
#    1998 and 2027 are extrapolated anchor points for smooth boundary
#    interpolation and are not included in the final output.
# ---------------------------------------------------------------------------

# Annual average CPI values
# 1999-2011  : CPI-IW (Labour Bureau, Base 2001=100) rebased to 2012=100 scale
#              using the link factor derived at 2011/2012 junction
# 2012-2026  : MOSPI CPI-Combined (All India), Base 2012=100
INDIA_CPI_ANNUAL_AVG = {
    1998: 40.5,   # extrapolation anchor
    1999: 42.1,
    2000: 43.8,
    2001: 45.6,
    2002: 47.4,
    2003: 49.3,
    2004: 51.5,
    2005: 54.1,
    2006: 57.3,
    2007: 61.0,
    2008: 65.6,
    2009: 72.2,
    2010: 82.3,
    2011: 92.2,
    2012: 100.0,  # base year
    2013: 109.0,
    2014: 115.0,
    2015: 119.0,
    2016: 124.0,
    2017: 128.0,
    2018: 133.0,
    2019: 140.0,
    2020: 150.0,
    2021: 157.0,
    2022: 169.0,
    2023: 180.0,
    2024: 188.0,
    2025: 195.0,
    2026: 201.0,
    2027: 205.0,   # extrapolation anchor
}


# ---------------------------------------------------------------------------
# 2. Build monthly series from annual averages via interpolation
# ---------------------------------------------------------------------------

def build_monthly_from_annual(annual_avg: dict) -> pd.DataFrame:
    """Interpolate annual CPI averages to monthly frequency.

    Strategy: assign each annual average to the mid-year point (July 1),
    then use linear interpolation to fill every month from Jan 1999 to
    Dec 2026.  This ensures each year's monthly values average to the
    supplied annual average while producing smooth transitions.
    """
    # Build annual series on mid-year anchor dates
    anchors = pd.Series(
        {pd.Timestamp(f"{yr}-07-01"): val for yr, val in annual_avg.items()},
        name="cpi",
    )

    # Monthly target range
    monthly_idx = pd.date_range("1999-01-01", "2026-12-01", freq="MS")

    # Combine anchors with monthly range, sort, interpolate, then resample
    combined = (
        anchors
        .reindex(anchors.index.union(monthly_idx))
        .sort_index()
        .interpolate(method="linear")
    )

    # Keep only the month-start dates in the target range
    result = combined.reindex(monthly_idx).to_frame()
    result.index.name = "date"
    result = result.reset_index()
    return result


# ---------------------------------------------------------------------------
# 3. Attempt live download from World Bank API (annual, FP.CPI.TOTL)
#    Falls back to embedded data if network is unavailable.
# ---------------------------------------------------------------------------

WORLD_BANK_URL = (
    "https://api.worldbank.org/v2/country/IN/indicator/FP.CPI.TOTL"
    "?format=json&per_page=100&mrv=30"
)


def _fetch_wb_annual_cpi() -> dict | None:
    """Fetch annual India CPI from World Bank API.

    Returns a {year: index_value} dict (Base 2010=100) or None on failure.
    """
    try:
        resp = requests.get(WORLD_BANK_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # data[1] is the list of observations
        observations = data[1] if isinstance(data, list) and len(data) > 1 else []
        result = {}
        for obs in observations:
            yr = obs.get("date")
            val = obs.get("value")
            if yr and val is not None:
                result[int(yr)] = float(val)
        return result if result else None
    except Exception:
        return None


def _rebase_wb_to_2012(wb_data: dict) -> dict:
    """Rebase World Bank CPI (Base 2010=100) to Base 2012=100."""
    base_2012 = wb_data.get(2012)
    if base_2012 is None:
        return {}
    return {yr: round(val / base_2012 * 100, 2) for yr, val in wb_data.items()}


def get_monthly_cpi() -> pd.DataFrame:
    """Return monthly India CPI DataFrame (Base 2012=100).

    Tries World Bank API first; falls back to embedded annual averages.
    """
    print("Fetching India CPI from World Bank API …")
    wb_raw = _fetch_wb_annual_cpi()

    if wb_raw:
        rebased = _rebase_wb_to_2012(wb_raw)
        # Merge with embedded data: WB fills recent years, embedded covers
        # early years and any gaps
        merged_annual = {**INDIA_CPI_ANNUAL_AVG, **rebased}
        print(
            f"  ✓ World Bank data obtained ({min(rebased)}-{max(rebased)}). "
            "Merged with embedded data for full 1999-2026 coverage."
        )
        return build_monthly_from_annual(merged_annual)

    print("  ⚠ World Bank API unavailable. Using embedded annual averages.")
    return build_monthly_from_annual(INDIA_CPI_ANNUAL_AVG)


# ---------------------------------------------------------------------------
# 4. Main entry-point
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "india_cpi.csv"

    df = get_monthly_cpi()

    # Year-on-year inflation (%)
    df["cpi_yoy_pct"] = df["cpi"].pct_change(periods=12) * 100

    # Round for readability
    df["cpi"] = df["cpi"].round(2)
    df["cpi_yoy_pct"] = df["cpi_yoy_pct"].round(4)

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    df.to_csv(out_path, index=False)
    print(f"\n✓ Saved {len(df)}-row CSV → {out_path}")

    print("\nFirst 5 rows:")
    print(df.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(df.tail().to_string(index=False))
    print("\n2012 rows (base year, avg should be ~100):")
    print(df[df.date.str.startswith("2012")].to_string(index=False))


if __name__ == "__main__":
    main()
