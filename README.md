# Roblox Bot Tester

**Purpose:**  
Test authenticated data scraping for Roblox creators using multiple account cookies, track rate‑limits, and output CSV for analysis.

## Setup

1. **Clone** this repo and `cd roblox-bot-tester`.
2. **Copy** `.env.example` → `.env` and fill in:
   - `TARGET_CREATORS`: comma‑separated user IDs
   - `ROBLOSECURITY_TOKENS`: comma‑separated `.ROBLOSECURITY` cookies (5 max)
  - `BATCH_SIZE`: how many universe IDs per API call (max 100)
   - `RATE_LIMIT_DELAY`: seconds to wait between requests (e.g. 0.7)
3. **Install** dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. **Run** one‑off scrape (writes to `test_data.csv`):
   ```bash
   python scraper.py
   ```
5. **Analyze** results:
   ```bash
   python analyze.py test_data.csv
   ```
6. **Deploy** on Railway:
   - Create new project, link this repo.
   - Add **PostgreSQL** plugin (if you later want DB).
   - Set environment variables from your `.env`.
   - Use **Jobs** to schedule `python scraper.py` every 30 min.

## Deliverables

- `test_data.csv`: raw scraped data  
- `reports/data_quality_report.md`: summary of missing/invalid fields  
- `reports/feasibility_report.md`: rate‑limit & account safety analysis  
