import pandas as pd
from pathlib import Path
from scripts.ingest_eurostat_csvs import tidy_save

def test_tidy_save_creates_files(tmpdir: Path):
    df = pd.DataFrame({
        "country": ["AT"],
        "year": [2015],
        "value": [1.0],
        "unit": ["x"]
    })

    (tmpdir / "data" / "processed").mkdir(parents=True, exist_ok=True)

    import scripts.ingest_eurostat_csvs as mod
    old_out = mod.OUT
    mod.OUT = tmpdir / "data" / "processed"

    try:
        tidy_save(df, "test_table")
        assert (mod.OUT / "test_table.csv").exists()
        assert (mod.OUT / "test_table.parquet").exists()
    finally:
        mod.OUT = old_out
