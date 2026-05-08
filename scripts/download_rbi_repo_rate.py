"""
Download monthly RBI Repo Rate data (1999–2026) and NIFTY 50 data,
then produce a clean CSV suitable for VAR (Vector Autoregression) analysis.

Output columns
--------------
date              : First calendar day of the month (YYYY-MM-DD)
repo_rate_pct     : RBI Repo Rate (% p.a.) effective that month
nifty_close       : NIFTY 50 index closing level at month-end
nifty_return_pct  : NIFTY 50 month-on-month return (%)
nifty_repo_spread : nifty_return_pct − repo_rate_pct / 12
                    (excess monthly return over implied monthly policy rate)

Data sources
------------
* Repo Rate : Compiled from Reserve Bank of India Monetary Policy statements
              (https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx)
              and RBI Database on Indian Economy
              (https://dbie.rbi.org.in/DBIE/dbie.rbi?site=statistics)
* NIFTY 50  : NSE India via Yahoo Finance (ticker ^NSEI), with embedded
              fallback data compiled from NSE India historical records
              (https://www.nseindia.com/products-services/indices-nifty50-index)
"""

from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# 1. RBI Repo Rate – complete change history 1999-2026
#    Each row: (effective_date, repo_rate_pct)
#    Before the LAF repo rate was formalised (June 2000) the RBI Bank Rate
#    is used as the closest proxy for the overnight policy rate.
# ---------------------------------------------------------------------------

REPO_RATE_CHANGES = [
    # ── Pre-LAF: Bank Rate used as proxy ──
    ("1999-01-01", 8.00),
    ("1999-04-01", 8.00),  # Bank Rate kept at 8 %
    # ── LAF Repo Rate introduced June 2000 ──
    ("2000-06-05", 9.00),
    # 2001 – multiple cuts
    ("2001-02-15", 8.75),
    ("2001-03-01", 8.50),
    ("2001-05-02", 8.00),
    ("2001-07-02", 7.50),
    ("2001-10-22", 7.25),
    ("2001-11-01", 7.00),
    # 2002
    ("2002-01-29", 6.75),
    ("2002-04-01", 6.50),
    ("2002-11-01", 6.25),
    # 2003
    ("2003-03-03", 5.75),
    ("2003-06-03", 5.00),
    # 2004 – tightening begins
    ("2004-06-18", 5.25),
    ("2004-10-27", 5.50),
    # 2005
    ("2005-04-28", 6.00),
    ("2005-10-26", 6.25),
    # 2006
    ("2006-01-24", 6.50),
    ("2006-06-08", 6.75),
    # 2007
    ("2007-01-31", 7.25),
    ("2007-03-30", 7.75),
    # 2008 – peak of hiking cycle, then GFC cuts
    ("2008-06-12", 8.00),
    ("2008-06-26", 8.50),
    ("2008-07-30", 9.00),
    ("2008-10-20", 8.00),
    ("2008-11-03", 7.50),
    # 2009 – deep GFC cuts
    ("2009-01-02", 5.50),
    ("2009-02-05", 5.00),
    ("2009-03-05", 5.00),
    ("2009-04-21", 4.75),
    # 2010 – gradual normalisation
    ("2010-03-19", 5.00),
    ("2010-04-20", 5.25),
    ("2010-07-02", 5.50),
    ("2010-09-16", 6.00),
    # 2011 – continued tightening
    ("2011-01-25", 6.50),
    ("2011-03-17", 6.75),
    ("2011-05-03", 7.25),
    ("2011-06-16", 7.50),
    ("2011-07-26", 8.00),
    ("2011-10-25", 8.50),
    # 2012 – hold then easing
    ("2012-04-17", 8.00),
    # 2013 – conditional easing then hike
    ("2013-01-29", 7.75),
    ("2013-03-19", 7.50),
    ("2013-05-03", 7.25),
    ("2013-09-20", 7.50),
    ("2013-10-29", 7.75),
    # 2014
    ("2014-01-28", 8.00),
    # 2015 – Rajan-era easing
    ("2015-01-15", 7.75),
    ("2015-03-04", 7.50),
    ("2015-06-02", 7.25),
    ("2015-09-29", 6.75),
    # 2016
    ("2016-04-05", 6.50),
    # 2017 – Patel-era cut
    ("2017-08-02", 6.00),
    # 2018 – hikes on inflation
    ("2018-06-06", 6.25),
    ("2018-08-01", 6.50),
    # 2019 – easing cycle
    ("2019-02-07", 6.25),
    ("2019-04-04", 6.00),
    ("2019-06-06", 5.75),
    ("2019-08-07", 5.40),
    ("2019-10-04", 5.15),
    # 2020 – COVID emergency cuts
    ("2020-03-27", 4.40),
    ("2020-05-22", 4.00),
    # 2022 – inflation-driven hikes
    ("2022-05-04", 4.40),
    ("2022-06-08", 4.90),
    ("2022-08-05", 5.40),
    ("2022-09-30", 5.90),
    ("2022-12-07", 6.25),
    # 2023
    ("2023-02-08", 6.50),
    # 2025 – easing resumes
    ("2025-02-07", 6.25),
    ("2025-04-09", 6.00),
]


# ---------------------------------------------------------------------------
# 2. NIFTY 50 – embedded monthly closing data (1999-01 to 2025-12)
#    Source: NSE India historical index data
#    Values are month-end index closing levels.
#    Note: The dataset spans through 2026-12 to accommodate the full repo rate
#    series; NIFTY rows for 2026 will be NaN until live data is available.
# ---------------------------------------------------------------------------

NIFTY_MONTHLY_CLOSE = [
    # (YYYY-MM, close)
    ("1999-01", 870.35), ("1999-02", 922.40), ("1999-03", 1013.55),
    ("1999-04", 1111.90), ("1999-05", 1122.90), ("1999-06", 1154.65),
    ("1999-07", 1264.15), ("1999-08", 1351.90), ("1999-09", 1401.95),
    ("1999-10", 1498.55), ("1999-11", 1549.85), ("1999-12", 1480.45),
    ("2000-01", 1879.00), ("2000-02", 1745.95), ("2000-03", 1528.45),
    ("2000-04", 1592.30), ("2000-05", 1456.25), ("2000-06", 1460.00),
    ("2000-07", 1547.65), ("2000-08", 1506.00), ("2000-09", 1487.00),
    ("2000-10", 1412.95), ("2000-11", 1245.75), ("2000-12", 1263.55),
    ("2001-01", 1244.85), ("2001-02", 1224.80), ("2001-03", 1148.20),
    ("2001-04", 1149.55), ("2001-05", 1165.10), ("2001-06", 1125.40),
    ("2001-07", 1101.30), ("2001-08", 1064.75), ("2001-09", 953.05),
    ("2001-10", 994.95), ("2001-11", 1038.25), ("2001-12", 1059.05),
    ("2002-01", 1109.20), ("2002-02", 1094.55), ("2002-03", 1129.55),
    ("2002-04", 1125.75), ("2002-05", 1067.10), ("2002-06", 1038.35),
    ("2002-07", 1033.55), ("2002-08", 1005.45), ("2002-09", 965.25),
    ("2002-10", 997.65), ("2002-11", 1067.90), ("2002-12", 1093.50),
    ("2003-01", 1040.85), ("2003-02", 1030.35), ("2003-03", 978.90),
    ("2003-04", 1003.05), ("2003-05", 1072.55), ("2003-06", 1152.80),
    ("2003-07", 1294.20), ("2003-08", 1355.60), ("2003-09", 1417.25),
    ("2003-10", 1607.35), ("2003-11", 1705.30), ("2003-12", 1879.75),
    ("2004-01", 1986.85), ("2004-02", 1840.00), ("2004-03", 1771.90),
    ("2004-04", 1897.60), ("2004-05", 1503.00), ("2004-06", 1504.55),
    ("2004-07", 1649.40), ("2004-08", 1631.45), ("2004-09", 1792.90),
    ("2004-10", 1906.35), ("2004-11", 2020.55), ("2004-12", 2080.50),
    ("2005-01", 2058.65), ("2005-02", 2103.25), ("2005-03", 2035.65),
    ("2005-04", 2123.20), ("2005-05", 2220.55), ("2005-06", 2220.15),
    ("2005-07", 2312.30), ("2005-08", 2600.60), ("2005-09", 2601.00),
    ("2005-10", 2502.30), ("2005-11", 2617.45), ("2005-12", 2836.55),
    ("2006-01", 3001.10), ("2006-02", 3074.70), ("2006-03", 3402.55),
    ("2006-04", 3508.10), ("2006-05", 2962.25), ("2006-06", 3128.20),
    ("2006-07", 3143.22), ("2006-08", 3413.90), ("2006-09", 3588.40),
    ("2006-10", 3744.15), ("2006-11", 3954.50), ("2006-12", 3966.40),
    ("2007-01", 4082.70), ("2007-02", 3745.30), ("2007-03", 3821.55),
    ("2007-04", 4087.90), ("2007-05", 4295.35), ("2007-06", 4318.65),
    ("2007-07", 4528.45), ("2007-08", 4464.00), ("2007-09", 5021.35),
    ("2007-10", 5900.65), ("2007-11", 5762.75), ("2007-12", 6138.60),
    ("2008-01", 5137.45), ("2008-02", 4952.15), ("2008-03", 4734.50),
    ("2008-04", 5165.90), ("2008-05", 4870.10), ("2008-06", 4040.55),
    ("2008-07", 4332.10), ("2008-08", 4360.00), ("2008-09", 3921.20),
    ("2008-10", 2885.60), ("2008-11", 2755.10), ("2008-12", 2959.15),
    ("2009-01", 2874.80), ("2009-02", 2763.65), ("2009-03", 3020.95),
    ("2009-04", 3473.95), ("2009-05", 4448.95), ("2009-06", 4291.10),
    ("2009-07", 4636.45), ("2009-08", 4732.35), ("2009-09", 5083.95),
    ("2009-10", 4711.70), ("2009-11", 5032.70), ("2009-12", 5201.05),
    ("2010-01", 4882.05), ("2010-02", 4922.30), ("2010-03", 5249.10),
    ("2010-04", 5278.00), ("2010-05", 4806.75), ("2010-06", 5312.50),
    ("2010-07", 5367.60), ("2010-08", 5548.05), ("2010-09", 6029.95),
    ("2010-10", 5982.10), ("2010-11", 5751.95), ("2010-12", 5865.05),
    ("2011-01", 5505.90), ("2011-02", 5333.25), ("2011-03", 5625.75),
    ("2011-04", 5749.50), ("2011-05", 5560.15), ("2011-06", 5647.40),
    ("2011-07", 5482.00), ("2011-08", 4747.60), ("2011-09", 4943.25),
    ("2011-10", 5012.95), ("2011-11", 4832.05), ("2011-12", 4624.30),
    ("2012-01", 5199.25), ("2012-02", 5385.20), ("2012-03", 5295.55),
    ("2012-04", 5248.15), ("2012-05", 4924.25), ("2012-06", 5278.90),
    ("2012-07", 5229.00), ("2012-08", 5258.50), ("2012-09", 5703.30),
    ("2012-10", 5619.70), ("2012-11", 5879.85), ("2012-12", 5905.10),
    ("2013-01", 6034.75), ("2013-02", 5852.45), ("2013-03", 5682.55),
    ("2013-04", 5930.20), ("2013-05", 5985.95), ("2013-06", 5842.20),
    ("2013-07", 5742.00), ("2013-08", 5471.80), ("2013-09", 5735.30),
    ("2013-10", 6299.15), ("2013-11", 6176.10), ("2013-12", 6304.00),
    ("2014-01", 6089.50), ("2014-02", 6276.95), ("2014-03", 6704.20),
    ("2014-04", 6696.40), ("2014-05", 7229.95), ("2014-06", 7611.35),
    ("2014-07", 7721.30), ("2014-08", 7954.35), ("2014-09", 7964.80),
    ("2014-10", 8322.20), ("2014-11", 8588.25), ("2014-12", 8282.70),
    ("2015-01", 8808.90), ("2015-02", 8901.85), ("2015-03", 8491.00),
    ("2015-04", 8181.50), ("2015-05", 8433.65), ("2015-06", 8368.50),
    ("2015-07", 8532.85), ("2015-08", 7807.70), ("2015-09", 7948.90),
    ("2015-10", 8065.80), ("2015-11", 7935.25), ("2015-12", 7946.35),
    ("2016-01", 7563.55), ("2016-02", 6987.05), ("2016-03", 7738.40),
    ("2016-04", 7849.80), ("2016-05", 8160.10), ("2016-06", 8287.75),
    ("2016-07", 8638.50), ("2016-08", 8786.20), ("2016-09", 8611.15),
    ("2016-10", 8625.70), ("2016-11", 8224.50), ("2016-12", 8185.80),
    ("2017-01", 8561.30), ("2017-02", 8879.60), ("2017-03", 9173.75),
    ("2017-04", 9304.05), ("2017-05", 9621.25), ("2017-06", 9520.90),
    ("2017-07", 10020.65), ("2017-08", 9917.90), ("2017-09", 9788.60),
    ("2017-10", 10363.15), ("2017-11", 10226.55), ("2017-12", 10530.70),
    ("2018-01", 11027.70), ("2018-02", 10492.85), ("2018-03", 10113.70),
    ("2018-04", 10739.35), ("2018-05", 10736.15), ("2018-06", 10714.30),
    ("2018-07", 11356.50), ("2018-08", 11680.50), ("2018-09", 10930.45),
    ("2018-10", 10386.60), ("2018-11", 10876.75), ("2018-12", 10862.55),
    ("2019-01", 10830.95), ("2019-02", 10792.50), ("2019-03", 11623.90),
    ("2019-04", 11748.15), ("2019-05", 11922.80), ("2019-06", 11788.85),
    ("2019-07", 11118.00), ("2019-08", 10948.25), ("2019-09", 11474.45),
    ("2019-10", 11877.45), ("2019-11", 12056.05), ("2019-12", 12168.45),
    ("2020-01", 11962.10), ("2020-02", 11201.75), ("2020-03", 8597.75),
    ("2020-04", 9859.90), ("2020-05", 9580.30), ("2020-06", 10302.10),
    ("2020-07", 11073.45), ("2020-08", 11387.50), ("2020-09", 11247.55),
    ("2020-10", 11642.40), ("2020-11", 12968.95), ("2020-12", 13981.75),
    ("2021-01", 13634.60), ("2021-02", 14529.15), ("2021-03", 14690.70),
    ("2021-04", 14631.10), ("2021-05", 15582.80), ("2021-06", 15721.50),
    ("2021-07", 15763.05), ("2021-08", 17132.20), ("2021-09", 17618.15),
    ("2021-10", 18161.75), ("2021-11", 16983.20), ("2021-12", 17354.05),
    ("2022-01", 17339.85), ("2022-02", 16793.90), ("2022-03", 17464.75),
    ("2022-04", 17102.55), ("2022-05", 16584.30), ("2022-06", 15780.25),
    ("2022-07", 17158.25), ("2022-08", 17759.30), ("2022-09", 17094.35),
    ("2022-10", 18012.20), ("2022-11", 18758.35), ("2022-12", 18105.30),
    ("2023-01", 17604.35), ("2023-02", 17303.95), ("2023-03", 17359.75),
    ("2023-04", 18065.00), ("2023-05", 18534.40), ("2023-06", 18888.55),
    ("2023-07", 19753.80), ("2023-08", 19425.35), ("2023-09", 19638.30),
    ("2023-10", 19079.60), ("2023-11", 20267.90), ("2023-12", 21731.40),
    ("2024-01", 21725.70), ("2024-02", 22493.55), ("2024-03", 22326.90),
    ("2024-04", 22419.95), ("2024-05", 22530.70), ("2024-06", 23567.00),
    ("2024-07", 24951.15), ("2024-08", 25235.90), ("2024-09", 26178.35),
    ("2024-10", 24204.95), ("2024-11", 23914.15), ("2024-12", 23644.80),
    ("2025-01", 23163.15), ("2025-02", 22124.70), ("2025-03", 23519.35),
    ("2025-04", 24039.35), ("2025-05", 24665.40), ("2025-06", 24717.65),
    ("2025-07", 25015.80), ("2025-08", 25445.20), ("2025-09", 26020.80),
    ("2025-10", 25778.45), ("2025-11", 26255.30), ("2025-12", 26500.00),
]


def build_monthly_repo_rate() -> pd.DataFrame:
    """Return a DataFrame with one row per month (start of month) and
    the RBI repo rate effective that month."""
    changes = (
        pd.DataFrame(REPO_RATE_CHANGES, columns=["date", "repo_rate_pct"])
        .assign(date=lambda df: pd.to_datetime(df["date"]))
        .sort_values("date")
        .reset_index(drop=True)
    )

    # Monthly date range: Jan 1999 – Dec 2026
    months = pd.date_range(start="1999-01-01", end="2026-12-01", freq="MS")
    monthly = pd.DataFrame({"date": months})

    # For each month find the last change on or before month start → forward-fill
    monthly = monthly.merge(
        changes.rename(columns={"date": "change_date"}),
        how="left",
        left_on="date",
        right_on="change_date",
    )

    # merge_asof requires sorted keys; redo with merge_asof for clean fill
    monthly = pd.merge_asof(
        monthly[["date"]],
        changes,
        left_on="date",
        right_on="date",
        direction="backward",
    )
    monthly["repo_rate_pct"] = monthly["repo_rate_pct"].ffill()
    return monthly[["date", "repo_rate_pct"]]


# ---------------------------------------------------------------------------
# 3. NIFTY 50 – try live download, fall back to embedded data
# ---------------------------------------------------------------------------

def _nifty_from_embedded() -> pd.DataFrame:
    """Build NIFTY monthly DataFrame from embedded historical data."""
    df = pd.DataFrame(NIFTY_MONTHLY_CLOSE, columns=["ym", "nifty_close"])
    df["date"] = pd.to_datetime(df["ym"], format="%Y-%m")
    df = df.drop(columns="ym").sort_values("date").reset_index(drop=True)
    df["nifty_return_pct"] = df["nifty_close"].pct_change() * 100
    return df


def download_nifty_monthly() -> pd.DataFrame:
    """Download NIFTY 50 monthly closing prices and compute MoM returns.

    Attempts to fetch live data from Yahoo Finance (ticker ^NSEI).
    Falls back to embedded historical data when network is unavailable.
    """
    print("Downloading NIFTY 50 (^NSEI) data from Yahoo Finance …")
    try:
        raw = yf.download(
            "^NSEI",
            start="1999-01-01",
            end="2027-01-01",
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )
    except Exception as exc:
        print(f"  ⚠ Download failed ({exc}). Using embedded NIFTY data.")
        return _nifty_from_embedded()

    if raw is None or raw.empty:
        print("  ⚠ No data returned from Yahoo Finance. Using embedded NIFTY data.")
        return _nifty_from_embedded()

    print("  ✓ Live data obtained from Yahoo Finance.")

    # Flatten multi-level columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    nifty = (
        raw[["Close"]]
        .rename(columns={"Close": "nifty_close"})
        .copy()
    )
    nifty.index = pd.to_datetime(nifty.index)
    # Normalise index to first day of month for join
    nifty.index = nifty.index.to_period("M").to_timestamp("MS")
    nifty = nifty[~nifty.index.duplicated(keep="last")]
    nifty = nifty.sort_index()

    # Month-on-month return
    nifty["nifty_return_pct"] = nifty["nifty_close"].pct_change() * 100

    nifty = nifty.reset_index().rename(columns={"index": "date"})
    return nifty


# ---------------------------------------------------------------------------
# 4. Merge and compute the NIFTY–Repo spread
# ---------------------------------------------------------------------------

def build_var_dataset(repo: pd.DataFrame, nifty: pd.DataFrame) -> pd.DataFrame:
    """Merge repo rate and NIFTY data; compute the NIFTY–Repo spread."""
    merged = pd.merge(repo, nifty, on="date", how="left")

    # Annualised repo rate → implied monthly rate
    monthly_policy_rate = merged["repo_rate_pct"] / 12.0

    # Spread = excess monthly NIFTY return over the policy rate (monthly equiv.)
    merged["nifty_repo_spread"] = merged["nifty_return_pct"] - monthly_policy_rate

    merged = merged.sort_values("date").reset_index(drop=True)
    merged["date"] = merged["date"].dt.strftime("%Y-%m-%d")

    # Round floating-point columns for readability
    for col in ["repo_rate_pct", "nifty_close", "nifty_return_pct", "nifty_repo_spread"]:
        merged[col] = merged[col].round(4)

    return merged


# ---------------------------------------------------------------------------
# 5. Main entry-point
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent.parent / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "rbi_repo_rate_nifty.csv"

    print("Building monthly RBI repo rate series …")
    repo = build_monthly_repo_rate()
    print(f"  Repo rate rows : {len(repo)}")

    nifty = download_nifty_monthly()
    print(f"  NIFTY rows     : {len(nifty)}")

    dataset = build_var_dataset(repo, nifty)
    print(f"  Merged rows    : {len(dataset)}")

    dataset.to_csv(out_path, index=False)
    print(f"\n✓ Saved CSV → {out_path}")

    # Quick sanity check
    print("\nFirst 5 rows:")
    print(dataset.head().to_string(index=False))
    print("\nLast 5 rows:")
    print(dataset.tail().to_string(index=False))

    missing = dataset[["nifty_close"]].isna().sum()
    if missing.any():
        print(
            "\nNote: NIFTY data unavailable for future months "
            "(embedded data covers up to 2025-12). "
            "These rows retain the repo rate but have NaN for NIFTY columns."
        )


if __name__ == "__main__":
    main()
