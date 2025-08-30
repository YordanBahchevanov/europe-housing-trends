from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

import scripts.analysis_utils as au


def test_per_country_block_monkeypatched_corr(monkeypatch):
    def fake_corr(df, x, y):
        return pd.DataFrame({
            "country": ["A", "B"],
            "pearson": [0.9, 0.1],
            "spearman": [0.8, 0.2],
            "N": [10, 8],
        })
    monkeypatch.setattr(au, "per_country_corr", fake_corr)

    df = pd.DataFrame({"country": ["A","A","B","B"], "x":[1,2,3,4], "y":[2,3,4,5]})
    out = au.per_country_block(df, "x", "y", label="hpi_vs_net")
    assert list(out.columns) == ["country", "hpi_vs_net_pearson", "hpi_vs_net_spearman", "N"]
    assert out.shape == (2, 4)
    assert np.isclose(out.loc[out["country"]=="A","hpi_vs_net_pearson"].iloc[0], 0.9)


def test_summarize_country():
    df = pd.DataFrame({
        "country": ["Bulgaria","A","B"],
        "hpi_vs_net_pearson": [0.99, 0.5, 0.7],
        "hpi_vs_net_spearman": [1.0, 0.6, 0.8],
        "N": [10, 9, 8],
    })
    row, eu_mean, eu_median = au.summarize_country(df, country="Bulgaria")
    assert row["country"].iloc[0] == "Bulgaria"
    assert "EU_mean" in eu_mean.index
    assert "EU_median" in eu_median.index
    assert "country" not in eu_mean.columns and "N" not in eu_mean.columns


def test_scatter_with_line_saves(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(au, "FIGS", tmp_path)
    df = pd.DataFrame({"x": [1,2,3,4], "y": [1,2,2,3]})
    au.scatter_with_line(df, x="x", y="y",
                         title="t", xlabel="x", ylabel="y",
                         outname="test_scatter.png")
    assert (tmp_path / "test_scatter.png").exists()


def test_make_affordability_columns():
    df = pd.DataFrame({
        "country": ["BG","BG","BG"],
        "year": [2015,2016,2017],
        "house_price_index": [100, 110, 120],
        "net_earnings": [10000, 11000, 12000],
        "real_earnings": [10000, 10500, 11500],
    })
    out = au.make_affordability_columns(df)
    assert {"afford_ratio","afford_ratio_real","afford_ratio_index","afford_ratio_real_index"} <= set(out.columns)
    sub = out[out["year"].isin([2015])]
    assert np.isclose(sub["afford_ratio_index"].iloc[0], 100.0)


def test_eu_average_series_basic():
    df = pd.DataFrame({
        "country":["A","A","B","B"],
        "year":[2015,2016,2015,2016],
        "afford_ratio_index":[100,110,100,120]
    })
    EUSET = {"A","B"}
    out = au.eu_average_series(df, EUSET, "afford_ratio_index")
    assert list(out["year"]) == [2015,2016]
    assert np.allclose(out[f"eu_afford_ratio_index_mean"], [100,115])


def test_plot_country_vs_eu_saves(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(au, "FIGS", tmp_path)
    bg = pd.DataFrame({"year":[2015,2016], "bg_aff_index":[100, 95]})
    eu = pd.DataFrame({"year":[2015,2016], "eu_afford_ratio_index_mean":[100, 110]})
    au.plot_country_vs_eu(bg, eu, "bg_aff_index", "eu_afford_ratio_index_mean",
                          title="BG vs EU", ylabel="idx", outname="bg_vs_eu.png")
    assert (tmp_path / "bg_vs_eu.png").exists()


def test_change_between():
    df = pd.DataFrame({"year":[2015,2016,2017], "val":[100,110,130]})
    assert au.change_between(df.rename(columns={"val":"val"}), "val", 2015, 2017) == 30.0
    assert np.isnan(au.change_between(df, "val", 2014, 2017))


def test_make_ratio_matrix_and_fill():
    df = pd.DataFrame({
        "country":["X","X","Y","Y","Y"],
        "year":[2015,2017,2015,2016,2017],
        "afford_ratio_index":[100,120,100,110,np.nan]
    })
    years = [2015,2016,2017]
    mat = au.make_ratio_matrix(df, "afford_ratio_index", years)
    assert mat.shape == (2,3)
    assert np.isclose(mat.loc["X", 2016], 110.0)
    assert np.isclose(mat.loc["Y", 2017], 110.0)


def test_kmeans_cluster_timecurves_shapes(monkeypatch):
    mat = pd.DataFrame({
        2015:[100,100,120],
        2016:[110, 90,130],
        2017:[120, 85,140],
    }, index=["A","B","C"])
    labels, centers = au.kmeans_cluster_timecurves(mat, k=2, random_state=42)
    assert set(labels.index) == {"A","B","C"}
    assert centers.shape == (2,3)


def test_plot_clusters_with_country_saves(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(au, "FIGS", tmp_path)
    mat = pd.DataFrame({
        2015:[100,100,120],
        2016:[110, 90,130],
        2017:[120, 85,140],
    }, index=["A","B","Bulgaria"])
    labels = pd.Series([0,0,1], index=mat.index, name="cluster")
    centers = pd.DataFrame([[ -0.5, 0.0, 0.5],
                            [  0.2, 0.1, -0.3]], columns=mat.columns)
    au.plot_clusters_with_country(mat, labels, centers,
                                  country_to_highlight="Bulgaria",
                                  title="Clusters", outname="clusters.png")
    assert (tmp_path / "clusters.png").exists()


def test_region_average_series():
    df = pd.DataFrame({
        "region":["East","East","West","West"],
        "year":[2015,2016,2015,2016],
        "afford_ratio_index":[100,110,100,120]
    })
    out = au.region_average_series(df, "afford_ratio_index")
    assert out.shape == (4,3)
    assert set(out.columns) == {"region","year","avg_value"}


def test_latest_extremes_vs_bg():
    pivot = pd.DataFrame({
        2023:[90,80,150],
        2024:[95,70,160],
    }, index=["Bulgaria","Finland","Hungary"])
    tup = au.latest_extremes_vs_bg(pivot)
    latest_year, min_ctry, min_val, max_ctry, max_val, pct_low, pct_high = tup
    assert latest_year == 2024
    assert min_ctry == "Finland" and max_ctry == "Hungary"
    assert pct_low > 0 and pct_high > 0


def test_summarize_afford_changes():
    bg_eu = pd.DataFrame({
        "year":[2015,2024],
        "bg_aff_index":[100, 89],
        "eu_afford_ratio_index_mean":[100, 112],
    })
    out = au.summarize_afford_changes(bg_eu)
    assert np.isclose(out["bg_change"], -11.0)
    assert np.isclose(out["eu_change"], 12.0)


def test_extract_bg_vs_eu_corr_and_helpers():
    per_geo = pd.DataFrame({
        "country":["Bulgaria","A","B"],
        "hpi_vs_net_pearson":[0.99,0.5,0.7],
        "hpi_vs_real_pearson":[0.97,0.4,0.6],
        "N":[10,9,8],
    })
    bg_row, eu_mean, eu_median = au.extract_bg_vs_eu_corr(per_geo)
    assert bg_row["country"].iloc[0] == "Bulgaria"
    assert "EU_mean" in eu_mean.index and "EU_median" in eu_median.index

    pooled = pd.DataFrame({
        "pair":["a","b","c"],
        "pearson":[0.2,-0.8,0.5],
        "spearman":[0.1,-0.7,0.4],
        "N":[100,100,100],
    })
    view = au.pooled_eu_context(pooled)
    assert view.iloc[0]["pair"] == "b"

    labels_df = pd.DataFrame({"country":["A","Bulgaria"], "cluster":[1,2]})
    assert au.get_bg_cluster_label(labels_df) == 2
    
    assert au.fmt(12.3456, digits=2) == "12.35"
    assert au.fmt(50, digits=1, pct=True) == "50.0%"
    assert au.fmt(np.nan) == "N/A"
    assert au.safe_get(pd.DataFrame({"x":[3.14]}), "x") == 3.14
    assert au.safe_get(pd.DataFrame({"y":[1]}), "x") is None
