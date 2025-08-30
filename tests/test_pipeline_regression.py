import pandas as pd
from pathlib import Path

def test_processed_panel_integrity():
    p = Path("data/processed/merged_panel.csv")
    assert p.exists(), "Run scripts/ingest_eurostat_csvs.py first"
    df = pd.read_csv(p)

    for col in ["country","year","house_price_index","net_earnings","unemployment_rate","hicp_index","gdp_per_capita","real_earnings"]:
        assert col in df.columns

    assert df[["house_price_index","net_earnings","hicp_index"]].isna().sum().sum() == 0
    bg = df[df["country"]=="Bulgaria"]
    assert (bg["year"].min() <= 2015) and (bg["year"].max() >= 2024)
