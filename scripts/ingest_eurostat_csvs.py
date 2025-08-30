from __future__ import annotations
from pathlib import Path
from typing import Optional, Mapping, Sequence
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)


def read_eurostat_clean_csv(
    path: Path,
    dim_filters: Optional[Mapping[str, str]] = None,
    keep_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Read a *clean* Eurostat Data Browser CSV and return a tidy DataFrame.

    Assumes standard columns like:
      DATAFLOW, LAST UPDATE, freq, <dims...>, geo, TIME_PERIOD, OBS_VALUE, [OBS_FLAG, CONF_STATUS]
    """
    df = pd.read_csv(path, sep=None, engine="python")

    df.columns = [c.strip().lower() for c in df.columns]

    rename_map = {}
    if "geo" in df.columns:
        rename_map["geo"] = "country"
    if "time_period" in df.columns:
        rename_map["time_period"] = "year"
    if "obs_value" in df.columns:
        rename_map["obs_value"] = "value"
    df = df.rename(columns=rename_map)

    if dim_filters:
        for k, v in dim_filters.items():
            if k in df.columns:
                df = df[df[k] == v]

    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    if "value" in df.columns:
        df["value"] = pd.to_numeric(df["value"], errors="coerce")

    default_keep = ["country", "year", "value", "unit"]
    cols = keep_cols if keep_cols is not None else default_keep
    cols = [c for c in cols if c in df.columns]

    if not cols:
        raise ValueError(
            f"No requested columns found in {path.name}. "
            f"Available: {list(df.columns)}; requested keep: {keep_cols or default_keep}"
        )

    for c in ("country", "unit"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    return df[cols].copy()



def tidy_save(df: pd.DataFrame, name: str) -> None:
    """
    Save a tidy DataFrame to both Parquet and CSV in data/processed/.

    Args:
        df: DataFrame to save (already filtered/renamed).
        name: Basename without extension (e.g., 'unemployment_rate_tidy').
    """
    out_parq = OUT / f"{name}.parquet"
    out_csv = OUT / f"{name}.csv"
    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved {name} -> {out_parq.name}, {out_csv.name}")


def main() -> None:
    """
    Ingest the five datasets you downloaded, using explicit filters that match your samples:

    - HPI (PRC_HPI_A): Annual, purchase='Purchases of existing dwellings', unit like 'Annual average index, 2010=100'
    - Net earnings (EARN_NT_NET): Annual, currency='Euro', estruct='Net earning',
      ecase='One-earner couple with two children earning 100% of the average earning'
    - Unemployment (UNE_RT_A): Annual, sex='Total', age='From 15 to 74 years',
      unit='Percentage of population in the labour force'
    - HICP (PRC_HICP_AIND): Annual, coicop='All-items HICP', unit='Annual average index'
    - GDP per capita (NAMA_10_PC): Annual, na_item='Gross domestic product at market prices',
      unit='Current prices, euro per capita'

    The filters keep a single, comparable slice from each dataset, so merges on (country, year) are straightforward.
    """
    files = {
        # name -> (path, filters)
        "house_price_index_tidy": (
            RAW / "house_price_index.csv",
            {
                "freq": "Annual",
                "purchase": "Purchases of existing dwellings",
                "unit": "Annual average index, 2010=100",
            },
        ),
        "net_earnings_tidy": (
            RAW / "net_earnings.csv",
            {
                "freq": "Annual",
                "currency": "Euro",
                "estruct": "Net earning",
                "ecase": "One-earner couple with two children earning 100% of the average earning",
            },
        ),
        "unemployment_rate_tidy": (
            RAW / "unemployment_rate.csv",
            {
                "freq": "Annual",
                "sex": "Total",
                "age": "From 15 to 74 years",
                "unit": "Percentage of population in the labour force",
            },
        ),
        "inflation_hicp_tidy": (
            RAW / "inflation_hicp.csv",
            {
                "freq": "Annual",
                "coicop": "All-items HICP",
                "unit": "Annual average index",
            },
        ),
        "gdp_per_capita_tidy": (
            RAW / "gdp_per_capita.csv",
            {
                "freq": "Annual",
                "na_item": "Gross domestic product at market prices",
                "unit": "Current prices, euro per capita",
            },
        ),
    }

    for name, (path, filters) in files.items():
        if not path.exists():
            print(f"WARNING: missing {path.name}, skipping {name}")
            continue

        df = read_eurostat_clean_csv(
            path,
            dim_filters=filters,
            keep_cols=["country", "year", "value", "unit"],
        )

        if "unit" in df.columns and df["unit"].nunique() > 1:
            top = df["unit"].value_counts().idxmax()
            df = df[df["unit"] == top].copy()

        tidy_save(df, name)


if __name__ == "__main__":
    main()