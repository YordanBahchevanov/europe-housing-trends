# Europe Housing Trends

This project explores the relationship between **house price dynamics** and **economic fundamentals** across Europe, with a special focus on **Bulgaria**.

---

## Research Question

**Are house price increases in Bulgaria more strongly decoupled from net earnings than in the rest of Europe?**

### Hypotheses
- **H0 (null):**  
  The relationship between HPI and net earnings in Bulgaria is similar to the European average.

- **H1 (alternative):**  
  Bulgaria’s HPI growth is less explained by net earnings compared to the European average, suggesting reduced affordability and stronger speculative/inflationary dynamics.

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

```europe-housing-trends/
│
├── data/
│ ├── raw/ # Original Eurostat CSVs
│ └── processed/ # Cleaned & merged CSVs
│
├── docs/
│ ├── 01_research_question.md
│ └── 02_data_sources.md
│
├── notebooks/
│ └── 01_exploration.ipynb
│
├── scripts/
│ └── ingest_eurostat_csvs.py
│
├── reports/
│ └── figures/ # Plots generated for report
│
└── README.md
```

--

## How to Reproduce

1. Clone this repo and set up environment:
   ```bash
   git clone https://github.com/YordanBahchevanov/europe-housing-trends.git
   cd europe-housing-trends
   conda create -n europe-housing-trends python=3.12
   conda activate europe-housing-trends