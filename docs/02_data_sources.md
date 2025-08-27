# Data Sources

All datasets are from **Eurostat Data Browser** (https://ec.europa.eu/eurostat/databrowser).

Each CSV was downloaded (Annual frequency) and filtered to a single consistent measure.

---

## 1. House Price Index (HPI)
- **Eurostat Code:** PRC_HPI_A  
- **Filter used:**  
  - Freq = Annual  
  - Purchase = Purchases of existing dwellings  
  - Unit = Annual average index (2010=100 or 2015=100 depending on country)  
- **CSV:** `data/raw/house_price_index.csv`  
- **Tidy:** `data/processed/house_price_index_tidy.csv`  
- **Variable:** `house_price_index`

---

## 2. Net Earnings
- **Eurostat Code:** EARN_NT_NET  
- **Filter used:**  
  - Freq = Annual  
  - Currency = Euro  
  - estruct = Net earning  
  - ecase = One-earner couple with two children earning 100% of average earning  
- **CSV:** `data/raw/net_earnings.csv`  
- **Tidy:** `data/processed/net_earnings_tidy.csv`  
- **Variable:** `net_earnings`

---

## 3. Unemployment Rate
- **Eurostat Code:** UNE_RT_A  
- **Filter used:**  
  - Freq = Annual  
  - Age = 15–74 years  
  - Sex = Total  
  - Unit = % of labour force  
- **CSV:** `data/raw/unemployment_rate.csv`  
- **Tidy:** `data/processed/unemployment_rate_tidy.csv`  
- **Variable:** `unemployment_rate`

---

## 4. Inflation (HICP Index)
- **Eurostat Code:** PRC_HICP_AIND  
- **Filter used:**  
  - Freq = Annual  
  - COICOP = All-items HICP  
  - Unit = Annual average index  
- **CSV:** `data/raw/inflation_hicp.csv`  
- **Tidy:** `data/processed/inflation_hicp_tidy.csv`  
- **Variable:** `hicp_index`

---

## 5. GDP per capita
- **Eurostat Code:** NAMA_10_PC  
- **Filter used:**  
  - Freq = Annual  
  - NA_item = GDP at market prices  
  - Unit = Current prices, euro per capita  
- **CSV:** `data/raw/gdp_per_capita.csv`  
- **Tidy:** `data/processed/gdp_per_capita_tidy.csv`  
- **Variable:** `gdp_per_capita`