import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import os

os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.cluster import KMeans

plt.rcParams["figure.figsize"] = (8, 5)
plt.rcParams["axes.grid"] = True

PROC = Path("../data/processed")
FIGS = Path("../reports/figures")
DOCS = Path("../docs")

PROC, FIGS

EU27 = {
    "Austria","Belgium","Bulgaria","Croatia","Cyprus","Czechia","Denmark","Estonia","Finland",
    "France","Germany","Greece","Hungary","Ireland","Italy","Latvia","Lithuania","Luxembourg",
    "Malta","Netherlands","Poland","Portugal","Romania","Slovakia","Slovenia","Spain","Sweden"
}


EASTERN = {
    "Bulgaria","Romania","Poland","Hungary","Czechia","Slovakia",
    "Lithuania","Latvia","Estonia","Croatia","Slovenia"
}

WESTERN = {
    "Austria","Belgium","France","Germany","Netherlands","Luxembourg",
    "Ireland","Italy","Spain","Portugal","Denmark","Sweden","Finland","Norway"
}

BALKANS = {"Bulgaria","Romania","Croatia","Slovenia","Greece"}


def corr_pair(df: pd.DataFrame, x: str, y: str) -> Dict[str, float]:
    """
    Compute Pearson and Spearman correlations between two columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame containing the columns `x` and `y`.
    x : str
        Column name for the first variable.
    y : str
        Column name for the second variable.

    Returns
    -------
    Dict[str, float]
        A dictionary with keys:
        - "pearson": Pearson correlation (NaN if < 3 rows after dropping NA).
        - "spearman": Spearman correlation (NaN if < 3 rows after dropping NA).
        - "N": number of non-missing observations used (as float for easy CSV export).

    Notes
    -----
    - Rows with NA in either column are dropped.
    - Requires at least 3 observations to compute both statistics.
    """
    d = df[[x, y]].dropna()
    if d.shape[0] < 3:
        return {"pearson": float("nan"), "spearman": float("nan"), "N": float(d.shape[0])}
    return {
        "pearson": float(d[x].corr(d[y], method="pearson")),
        "spearman": float(d[x].corr(d[y], method="spearman")),
        "N": float(d.shape[0]),
    }


def per_country_corr(df: pd.DataFrame, x: str, y: str) -> pd.DataFrame:
    """
    Compute per-country Pearson and Spearman correlations between two columns.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame; must contain columns: 'country', `x`, `y`.
    x : str
        Column name for the first variable.
    y : str
        Column name for the second variable.

    Returns
    -------
    pd.DataFrame
        One row per country with columns:
        - country : str
        - pearson : float
        - spearman : float
        - N : int  (observations used per country)

    Notes
    -----
    - Drops rows with NA in either `x` or `y` within each country.
    - Countries with fewer than 3 observations are excluded.
    """
    rows: List[Dict[str, float]] = []
    for c, g in df.groupby("country", dropna=False):
        d = g[[x, y]].dropna()
        if d.shape[0] >= 3:
            rows.append({
                "country": c,
                "pearson": float(d[x].corr(d[y], method="pearson")),
                "spearman": float(d[x].corr(d[y], method="spearman")),
                "N": int(d.shape[0]),
            })
    return pd.DataFrame(rows).sort_values("country").reset_index(drop=True)


def per_country_block(df: pd.DataFrame, x: str, y: str, label: str) -> pd.DataFrame:
    """
    Wrap `per_country_corr` and rename outputs with a label prefix.

    Parameters
    ----------
    df : pd.DataFrame
        Input frame containing 'country', `x`, and `y`.
    x : str
        First variable name.
    y : str
        Second variable name.
    label : str
        Prefix for output columns (e.g., "hpi_vs_net").

    Returns
    -------
    pd.DataFrame
        Columns:
        - country : str
        - {label}_pearson : float
        - {label}_spearman : float
        - N : int
    """
    out = per_country_corr(df, x, y)
    out = out.rename(columns={
        "pearson": f"{label}_pearson",
        "spearman": f"{label}_spearman",
    })
    return out[["country", f"{label}_pearson", f"{label}_spearman", "N"]]


def summarize_country(df: pd.DataFrame, country: str = "Bulgaria") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract the row for a given country and compute EU mean/median across countries.

    Parameters
    ----------
    df : pd.DataFrame
        Per-country correlation table (must include 'country' and 'N' plus metric columns).
    country : str, default "Bulgaria"
        Country to extract for comparison.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - row : 1xK DataFrame for the selected country
        - eu_mean : 1xK DataFrame (mean across countries for numeric columns)
        - eu_median : 1xK DataFrame (median across countries for numeric columns)

    Notes
    -----
    - Non-numeric columns ('country', 'N') are excluded from mean/median.
    """
    row = df[df["country"] == country]
    eu_mean = df.drop(columns=["country", "N"]).mean(numeric_only=True).to_frame("EU_mean").T
    eu_median = df.drop(columns=["country", "N"]).median(numeric_only=True).to_frame("EU_median").T
    return row, eu_mean, eu_median


def scatter_with_line(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    outname: str,
) -> None:
    """
    Draw a scatter plot with a simple OLS best-fit line (via numpy.polyfit).

    Parameters
    ----------
    df : pd.DataFrame
        Input frame containing columns `x` and `y`.
    x : str
        X-axis variable name.
    y : str
        Y-axis variable name.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    outname : str
        File name (PNG) to save under `reports/figures/`.

    Returns
    -------
    None
        Saves the figure and displays it.

    Notes
    -----
    - Rows with NA in `x` or `y` are dropped.
    - If fewer than 2 valid points remain, only the scatter is drawn (no line).
    """
    d = df[[x, y]].dropna()
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(d[x], d[y], alpha=0.35)

    if len(d) >= 2:
        X = d[x].to_numpy()
        Y = d[y].to_numpy()
        mask = np.isfinite(X) & np.isfinite(Y)
        if mask.sum() >= 2:
            b1, b0 = np.polyfit(X[mask], Y[mask], 1)
            xs = np.linspace(X[mask].min(), X[mask].max(), 100)
            ax.plot(xs, b0 + b1 * xs, linewidth=2, label=f"OLS slope = {b1:.3g}")
            ax.legend()

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    plt.tight_layout()
    plt.savefig(FIGS / outname, dpi=160)
    plt.show()


def make_affordability_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create affordability ratio columns (levels and base-100 indices).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: 'country', 'year', 'house_price_index',
        'net_earnings', and 'real_earnings'.

    Returns
    -------
    pd.DataFrame
        Original columns plus:
        - 'afford_ratio'        = HPI / net_earnings
        - 'afford_ratio_real'   = HPI / real_earnings
        - 'afford_ratio_index'  = 100 * afford_ratio / afford_ratio[first_year_in_country]
        - 'afford_ratio_real_index' = 100 * afford_ratio_real / afford_ratio_real[first_year]
    """
    out = df.copy()
    out["afford_ratio"] = out["house_price_index"] / out["net_earnings"]
    out["afford_ratio_real"] = out["house_price_index"] / out["real_earnings"]

    def base100(s: pd.Series) -> pd.Series:
        base = s.iloc[0]
        return (s / base) * 100.0

    out["afford_ratio_index"] = (
        out.sort_values(["country","year"])
           .groupby("country", group_keys=False)["afford_ratio"]
           .apply(base100)
    )
    out["afford_ratio_real_index"] = (
        out.sort_values(["country","year"])
           .groupby("country", group_keys=False)["afford_ratio_real"]
           .apply(base100)
    )
    return out


def eu_average_series(
    df: pd.DataFrame,
    include_set: set,
    value_col: str
) -> pd.DataFrame:
    """
    Compute a simple (unweighted) EU average per-year for a column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'country', 'year', and `value_col`.
    include_set : set
        Set of country names to include (e.g., EU27).
    value_col : str
        Column to average (e.g., 'afford_ratio_index').

    Returns
    -------
    pd.DataFrame
        Columns: 'year', f'eu_{value_col}_mean'
    """
    tmp = df[df["country"].isin(include_set)]
    ser = (
        tmp.groupby("year")[value_col]
           .mean()
           .rename(f"eu_{value_col}_mean")
           .reset_index()
    )
    return ser


def plot_country_vs_eu(
    country_df: pd.DataFrame,
    eu_df: pd.DataFrame,
    country_col: str,
    eu_col: str,
    title: str,
    ylabel: str,
    outname: str
) -> None:
    """
    Plot a country series vs the EU mean series.

    Parameters
    ----------
    country_df : pd.DataFrame
        Columns: 'year', `country_col`
    eu_df : pd.DataFrame
        Columns: 'year', `eu_col`
    country_col : str
        Country value column name (e.g., 'bg_aff_index')
    eu_col : str
        EU value column name (e.g., 'eu_afford_ratio_index_mean')
    title : str
        Plot title
    ylabel : str
        Y-axis label
    outname : str
        File name for PNG under reports/figures/

    Returns
    -------
    None
    """
    d = country_df.merge(eu_df, on="year", how="inner").sort_values("year")
    fig, ax = plt.subplots(figsize=(8.5,5.5))
    ax.plot(d["year"], d[country_col], marker="o", linewidth=2.5, label="Bulgaria")
    ax.plot(d["year"], d[eu_col], marker="s", linewidth=2.5, label="EU average")
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend()
    plt.tight_layout()
    plt.savefig(FIGS / outname, dpi=160)
    plt.show()


def change_between(
    df: pd.DataFrame, value_col: str, start: int, end: int
) -> float:
    """
    Compute change of `value_col` between two years.

    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: 'year', `value_col`.
    value_col : str
        Column to compare (e.g., 'bg_aff_index').
    start : int
        Start year (inclusive).
    end : int
        End year (inclusive).

    Returns
    -------
    float
        value[end] - value[start]
    """
    d = df.set_index("year")[value_col]
    if start not in d.index or end not in d.index:
        return float("nan")
    return float(d.loc[end] - d.loc[start])


def make_ratio_matrix(df: pd.DataFrame, value_col: str, years: List[int]) -> pd.DataFrame:
    """
    Create a pivot matrix of value_col with countries as rows and years as columns.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'country', 'year', and value_col.
    value_col : str
        Column to pivot (e.g., 'afford_ratio_index').
    years : List[int]
        Ordered list of year columns to include.

    Returns
    -------
    pd.DataFrame
        Index = country, columns = years. Missing values are linearly interpolated
        within each country, then forward/back-filled if needed.
    """
    p = (
        df.pivot(index="country", columns="year", values=value_col)
          .reindex(columns=years)
          .sort_index()
    )
    p = p.apply(lambda s: s.interpolate(axis=0, limit_direction="both"), axis=1)
    p = p.ffill(axis=1).bfill(axis=1)
    return p


def kmeans_cluster_timecurves(
    matrix: pd.DataFrame,
    k: int = 3,
    random_state: int = 42
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Run KMeans on **z-scored per-country** time series.

    Parameters
    ----------
    matrix : pd.DataFrame
        Country × Year matrix of the series (e.g., afford_ratio_index).
    k : int, default 3
        Number of clusters.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        - labels : pd.Series mapping country -> cluster_id
        - centers : pd.DataFrame with cluster centers (in z-scored space)
    """
    Z = matrix.copy()
    Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0)
    Z = Z.fillna(0.0)

    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    km.fit(Z.values)

    labels = pd.Series(km.labels_, index=Z.index, name="cluster")
    centers = pd.DataFrame(km.cluster_centers_, columns=Z.columns)
    return labels, centers


def plot_clusters_with_country(
    matrix: pd.DataFrame,
    labels: pd.Series,
    centers: pd.DataFrame,
    country_to_highlight: str,
    title: str,
    outname: str
) -> None:
    """
    Plot cluster centers and all member trajectories (light), highlight one country.

    Parameters
    ----------
    matrix : pd.DataFrame
        Country × Year matrix (levels, not z-scored).
    labels : pd.Series
        Country -> cluster_id.
    centers : pd.DataFrame
        Cluster centers in z-scored space (same columns as matrix).
    country_to_highlight : str
        Country name to emphasize (e.g., 'Bulgaria').
    title : str
        Plot title.
    outname : str
        PNG filename under reports/figures/.

    Returns
    -------
    None
    """
    Z = matrix.copy()
    Z = Z.sub(Z.mean(axis=1), axis=0).div(Z.std(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(10,6))
    years = Z.columns.astype(int)

    for c_id in sorted(labels.unique()):
        members = labels[labels == c_id].index
        ax.plot(years, Z.loc[members].T, color="lightgray", alpha=0.5, linewidth=1)

    for c_id in range(centers.shape[0]):
        ax.plot(years, centers.iloc[c_id].values, linewidth=3, label=f"Cluster {c_id}")

    if country_to_highlight in Z.index:
        ax.plot(years, Z.loc[country_to_highlight].values, linewidth=3.5, color="red", label=country_to_highlight)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("z-score within country", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper left", ncol=2)
    plt.tight_layout()
    plt.savefig(FIGS / outname, dpi=160)
    plt.show()


def assign_region(country: str) -> str:
    if country in BALKANS:
        return "Balkans"
    elif country in EASTERN:
        return "Eastern EU"
    elif country in WESTERN:
        return "Western/Nordic"
    else:
        return "Other"


def region_average_series(df: pd.DataFrame, value_col: str = "afford_ratio_index") -> pd.DataFrame:
    """
    Compute average trajectory per region.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'region', 'year', and `value_col`.
    value_col : str
        Column to average (e.g., 'afford_ratio_index').

    Returns
    -------
    pd.DataFrame
        Columns: region, year, avg_value
    """
    out = (
        df.groupby(["region","year"])[value_col]
          .mean()
          .reset_index()
          .rename(columns={value_col: "avg_value"})
    )
    return out


def latest_extremes_vs_bg(
    pivot: pd.DataFrame,
    bg_name: str = "Bulgaria"
) -> Tuple[int, str, float, str, float, float, float]:
    """
    Find (in the latest available year) the country with the lowest and highest
    affordability ratio index and compute percentage differences vs Bulgaria.

    Parameters
    ----------
    pivot : pd.DataFrame
        Country × Year matrix of afford_ratio_index (base=100 per country).
    bg_name : str, default "Bulgaria"
        Country name for Bulgaria in the index.

    Returns
    -------
    Tuple[int, str, float, str, float, float, float]
        (latest_year,
         min_country, min_value,
         max_country, max_value,
         pct_lower_than_bg, pct_higher_than_bg)

        where:
        pct_lower_than_bg = (BG - MIN) / BG * 100
        pct_higher_than_bg = (MAX - BG) / BG * 100
    """
    latest_year = int(pivot.columns.max())
    series_latest = pivot[latest_year].dropna()

    if bg_name not in series_latest.index:
        raise ValueError(f"{bg_name} not found in affordability index for {latest_year}.")

    bg_val = float(series_latest.loc[bg_name])
    min_country = str(series_latest.idxmin())
    min_value = float(series_latest.min())
    max_country = str(series_latest.idxmax())
    max_value = float(series_latest.max())

    pct_lower_than_bg = (bg_val - min_value) / bg_val * 100.0 if bg_val != 0 else float("nan")
    pct_higher_than_bg = (max_value - bg_val) / bg_val * 100.0 if bg_val != 0 else float("nan")

    return (latest_year, min_country, min_value, max_country, max_value, pct_lower_than_bg, pct_higher_than_bg)


def plot_countries_and_eu_afford(
    panel_df: pd.DataFrame,
    countries: List[str],
    eu_series: pd.DataFrame,
    title: str,
    outname: str
) -> None:
    """
    Plot affordability ratio index for a set of countries and the EU average.

    Parameters
    ----------
    panel_df : pd.DataFrame
        Long panel with columns ['country','year','afford_ratio_index'].
    countries : List[str]
        List of country names to plot.
    eu_series : pd.DataFrame
        EU average series with columns ['year', 'eu_afford_ratio_index_mean'].
    title : str
        Plot title.
    outname : str
        PNG filename saved under reports/figures/.
    """
    fig, ax = plt.subplots(figsize=(9,6))

    for c in countries:
        d = (panel_df[panel_df["country"] == c]
             .sort_values("year")[["year","afford_ratio_index"]])
        if c == "Bulgaria":
            ax.plot(d["year"], d["afford_ratio_index"], linewidth=3, marker="o", label=c, color="red")
        else:
            ax.plot(d["year"], d["afford_ratio_index"], linewidth=2, marker="o", label=c)

    d_eu = eu_series.sort_values("year").rename(columns={"eu_afford_ratio_index_mean":"EU average"})
    ax.plot(d_eu["year"], d_eu["EU average"], linewidth=3, marker="s", linestyle="--", label="EU average")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax.set_ylabel("Index (first year = 100)", fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, frameon=False)

    plt.tight_layout()
    plt.savefig(FIGS / outname, dpi=160, bbox_inches="tight")
    plt.show()


def change_between_years(df: pd.DataFrame, year_col: str, value_col: str, start_year: int, end_year: int) -> Optional[float]:
    """
    Compute value[end_year] - value[start_year].

    Parameters
    ----------
    df : pd.DataFrame
        Data containing `year_col` and `value_col`.
    year_col : str
        Name of year column (e.g., 'year').
    value_col : str
        Name of value column (e.g., 'bg_aff_index').
    start_year : int
        Start year (inclusive).
    end_year : int
        End year (inclusive).

    Returns
    -------
    Optional[float]
        Difference or None if either year is missing.
    """
    s = df.set_index(year_col)[value_col]
    if start_year not in s.index or end_year not in s.index:
        return None
    return float(s.loc[end_year] - s.loc[start_year])


def summarize_afford_changes(bg_eu: pd.DataFrame) -> Dict[str, float]:
    """
    Compute Bulgaria and EU affordability ratio index changes (2015→2024).

    Parameters
    ----------
    bg_eu : pd.DataFrame
        Columns: 'year', 'bg_aff_index', 'eu_afford_ratio_index_mean'.

    Returns
    -------
    Dict[str, float]
        {'bg_change': float, 'eu_change': float}
    """
    out = {
        "bg_change": change_between_years(bg_eu, "year", "bg_aff_index", 2015, 2024),
        "eu_change": change_between_years(bg_eu, "year", "eu_afford_ratio_index_mean", 2015, 2024),
    }
    return out


def extract_bg_vs_eu_corr(per_geo: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract Bulgaria's per-country correlations and EU mean/median across countries.

    Parameters
    ----------
    per_geo : pd.DataFrame
        Per-country correlation table (from Notebook 03).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        (bg_row, eu_mean, eu_median)
    """
    bg_row = per_geo[per_geo["country"] == "Bulgaria"].copy()
    eu_mean = per_geo.drop(columns=["country","N"]).mean(numeric_only=True).to_frame("EU_mean").T
    eu_median = per_geo.drop(columns=["country","N"]).median(numeric_only=True).to_frame("EU_median").T
    return bg_row, eu_mean, eu_median


def pooled_eu_context(pooled: pd.DataFrame) -> pd.DataFrame:
    """
    Return pooled EU correlations for display, sorted by absolute Pearson strength.

    Parameters
    ----------
    pooled : pd.DataFrame
        Columns include 'pair', 'pearson', 'spearman', 'N'

    Returns
    -------
    pd.DataFrame
        Sorted view by |pearson| descending.
    """
    view = pooled.copy()
    view["abs_pearson"] = view["pearson"].abs()
    return view.sort_values("abs_pearson", ascending=False)[["pair","pearson","spearman","N"]]


def get_bg_cluster_label(labels_df: pd.DataFrame) -> Optional[int]:
    """
    Extract Bulgaria's cluster label from the clusters table.

    Parameters
    ----------
    labels_df : pd.DataFrame
        Columns: 'country', 'cluster'.

    Returns
    -------
    Optional[int]
        Cluster id or None if not found.
    """
    row = labels_df[labels_df["country"] == "Bulgaria"]
    if row.empty:
        return None
    return int(row["cluster"].iloc[0])


def fmt(x: Optional[float], digits: int = 2, pct: bool = False) -> str:
    """
    Format a number for human-readable text.

    Parameters
    ----------
    x : Optional[float]
        Value to format (may be None).
    digits : int
        Decimal places.
    pct : bool
        If True, append '%'.

    Returns
    -------
    str
        Formatted string.
    """
    if x is None or np.isnan(x):
        return "N/A"
    s = f"{x:.{digits}f}"
    return f"{s}%" if pct else s


def safe_get(df: pd.DataFrame, col: str) -> Optional[float]:
    return None if col not in df.columns or df.empty else float(df[col].iloc[0])
