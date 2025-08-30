from pathlib import Path
from scripts.ingest_eurostat_csvs import read_eurostat_clean_csv

def test_read_eurostat_csv_shapes(sample_hpi_csv: Path):
    df = read_eurostat_clean_csv(sample_hpi_csv)
    assert set(df.columns) >= {"country","year","value","unit"}
    assert len(df) == 2
    assert df["year"].dtype.name in ("Int64","int64")
