# Automated Trend Detection System (TrendPulse)

A real-time 4-step trend detection pipeline built in **Python 3.12** using **FastAPI**, **Pandas**, and **SQLite3**, featuring continuous multi-source data ingestion, NLP topic clustering, statistical rate-of-change velocity scoring, automated Discord/Slack webhook alerting, and Google Gemini AI trend intelligence briefs.

---

## 🏗️ 4-Step Pipeline Architecture

1. **Step 1: Ingestion**: Multi-source polling across Reddit (`/rising`), Google Trends RSS, Bluesky, Hacker News, Wikipedia Pageviews, and BBC Global Wire dispatches.
2. **Step 2: NLP Clustering**: Autonomous Named Entity Recognition (NER) for `ORG`, `PERSON`, `TECH`, `PRODUCT` and dynamic topical clustering.
3. **Step 3: Velocity Engine**: Statistical rate-of-change engine computing 1-hour volume ($V_{1h}$) vs 7-day rolling baselines ($\mu_{7d}$) and Z-scores using **pandas**.
4. **Step 4: Alerting & Webhooks**: Automated notification dispatch to Discord and Slack webhooks when velocity thresholds are breached, enriched with Gemini AI trend summaries.

---

## 🚀 How to Run Locally

### Prerequisites
- **Python 3.12+**
- **Git**

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/automated-trend-detection-system.git
cd automated-trend-detection-system
```

### 2. Set Up Virtual Environment & Dependencies

**Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

**macOS / Linux (Bash):**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. (Optional) Configure Gemini API Key
Copy the template and set your API key in `.env`:
```bash
cp .env.example .env
```
In `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(If no API key is provided, the engine automatically uses an intelligent algorithmic summary fallback.)*

### 4. Start the Application
```bash
uvicorn backend.main:app --reload --port 8000
```

Open **`http://127.0.0.1:8000`** in your browser to view the live dashboard!

---

## 📁 Repository Structure

```
automated-trend-detection-system/
├── backend/
│   ├── main.py               # FastAPI server, background worker, CORS, and REST endpoints
│   ├── models.py             # Pydantic data schemas
│   ├── database.py           # SQLite persistence for raw posts, clusters, and alerts
│   ├── sync_db.py            # Database synchronizer and cluster recalculation utility
│   ├── static/
│   │   └── index.html        # Modern reactive web dashboard
│   └── pipeline/
│       ├── ingestion.py      # Multi-source feed fetchers (Reddit, Trends, Bsky, HN, Wiki, Wire)
│       ├── nlp_extractor.py  # Named entity extraction & dynamic topic clustering
│       ├── velocity_engine.py# Rate-of-change scoring (pandas rolling window, Z-score)
│       ├── webhooks.py       # Discord embeds and Slack blocks webhook dispatcher
│       └── ai_summary.py     # Gemini 3.8 Flash trend brief generator (google-genai SDK)
├── .env.example              # Environment variables template
├── .gitignore                # Git ignore specifications
├── LICENSE                   # MIT License
├── metadata.json             # Project metadata
├── README.md                 # Project documentation
└── requirements.txt          # Python dependencies
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
