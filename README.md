# Europe Housing Trends

This project explores the relationship between **house price dynamics** and **economic fundamentals** across Europe, with a special focus on **Bulgaria**.

---

## Research Question

**Are house price increases in Bulgaria more strongly decoupled from net earnings than in the rest of Europe?**

### Hypotheses

- **H0 (null):**  
  Bulgaria’s housing affordability (HPI-to-earnings ratio) has evolved similarly to the European average.

- **H1 (alternative):**  
  Bulgaria’s housing affordability has diverged from the European average, showing a different trajectory compared to most EU countries.

---

## Data Sources

All datasets come from **Eurostat Data Browser**, exported as CSV and stored in `data/raw/`:

- **House Price Index (HPI)** — `house_price_index.csv`  
- **Net Earnings** — `net_earnings.csv`  
- **Unemployment Rate** — `unemployment_rate.csv`  
- **Inflation (HICP)** — `inflation_hicp.csv`  
- **GDP per capita** — `gdp_per_capita.csv`

Tidy, filtered versions are saved under `data/processed/`.

---

## Repository Structure

```
europe-housing-trends/
│
├── configs/
│   └── project.yaml     # Central configuration (paths, years, focus country)
│
├── data/
│   ├── raw/          # Original Eurostat CSVs
│   └── processed/    # Cleaned & merged CSVs
│
├── docs/             # Research documentation
│   ├── 01_research_question.md
│   ├── 02_data_sources.md
│   └── 03_conclusion.md
│
├── notebooks/        # Jupyter notebooks (analysis steps)
│   ├── 01_data_preparation.ipynb
│   ├── 02_exploratory_analysis.ipynb
│   ├── 03_correlation_analysis.ipynb
│   ├── 04_affordability_analysis.ipynb
│   └── 05_conclusion_and_report.ipynb
│
├── reports/
│   └── figures/      # Plots generated for report
│
├── scripts/          # Helper scripts
│   └── ingest_eurostat_csvs.py
│
├── environment.yml   # Reproducible conda environment
└── README.md
```

--

## How to Reproduce

1. Clone this repo:
  ```
  git clone https://github.com/YordanBahchevanov/europe-housing-trends.git
  cd europe-housing-trends
  ```
2. Create and activate the environment:
  ```
  conda env create -f environment.yml
  conda activate europe-housing-trends
  ```
3. Download the datasets from Eurostat and save them as CSVs in data/raw/ with exact filenames:
  - Create the folder if needed.
  - For details about the datasets, look [docs/02_data_sources.md](https://github.com/YordanBahchevanov/europe-housing-trends/blob/main/docs/02_data_sources.md)
4. Run the data ingestion script:
  ```
  python scripts/ingest_eurostat_csvs.py
  ```
5. Open and run notebooks step by step.

--

## Analysis Notebooks

- [01_data_preparation.ipynb](https://github.com/YordanBahchevanov/europe-housing-trends/blob/main/notebooks/01_data_preparation.ipynb) — Clean and merge Eurostat CSVs  
- [02_exploratory_analysis.ipynb](https://github.com/YordanBahchevanov/europe-housing-trends/blob/main/notebooks/02_exploratory_analysis.ipynb) — Europe-wide and Bulgaria-specific exploration  
- [03_correlation_analysis.ipynb](https://github.com/YordanBahchevanov/europe-housing-trends/blob/main/notebooks/03_correlation_analysis.ipynb) — Correlations between HPI, earnings, GDP, inflation, unemployment  
- [04_affordability_analysis.ipynb](https://github.com/YordanBahchevanov/europe-housing-trends/blob/main/notebooks/04_affordability_analysis.ipynb) — Affordability ratio, Bulgaria vs EU, clustering  
- [05_conclusion_and_report.ipynb](https://github.com/YordanBahchevanov/europe-housing-trends/blob/main/notebooks/05_conclusion_and_report.ipynb) — Final conclusions and policy implications

--

## Testing

This project uses **pytest** for unit testing.  
Tests cover both the ingestion pipeline (`scripts/ingest_eurostat_csvs.py`) and analysis helper functions from the notebooks.

To run all tests:

```bash
pytest -q
```

To run specific test file:

```bash
pytest tests/test_ingest_unit.py -v
```
--

## Key Results (2015–2024)

- Europe (EU pooled):
  - HPI correlates moderately with GDP growth and inflation.
  - Weak links with earnings; negative with unemployment.
- Bulgaria:
  - HPI is tightly coupled with net and real earnings.
  - Stronger fundamentals-driven pattern than the EU average.
  - Affordability improved (ratio index –11%), while EU average worsened (+12%).
- Extremes in 2024:
  - Lowest affordability index: Romania (64.1, ~27.9% lower than Bulgaria).
  - Highest affordability index: Hungary (161.7, ~81.7% higher than Bulgaria).