# مُرشِد / MSA — Shariah-Aware Quant Terminal

**خبيرك الكمّي المتوافق مع الشريعة** — Arabic-first, agentic market analysis platform.

MSA fuses nine live analytical modules, FinBERT news sentiment, a 4-pillar weighted composite score, GPT-4o judgment, Monte Carlo simulation, AAOIFI-style Shariah screening, and a halal-constrained portfolio optimizer into one Arabic-first quant terminal — built for Alinma's Shariah-compliant customer.

Live API: `https://vd2mi-msa.hf.space`

## Module Status — Live vs Beta vs Target

Credibility is the product. Every module is labeled by what it actually is:

| Module | Status | Notes |
|--------|--------|-------|
| 9 analytical modules (RSI-14, MACD, Bollinger, OBV, SG denoised trend, Z-Score, volume profile, F&G, SMA 50/200) | ✅ **Live** | Real math on real 1y OHLCV |
| FinBERT news sentiment | ✅ **Live** | ProsusAI/finbert via HF Inference API, keyword fallback |
| 4-pillar composite score + GPT-4o verdict | ✅ **Live** | 35% technical / 25% sentiment / 20% analyst / 20% volume |
| Arabic agentic copilot (`POST /chat`) | ✅ **Live** | GPT-4o tool-calling over the live endpoints; anti-hallucination rules — quotes only tool-returned numbers, states failures explicitly |
| Monte Carlo simulator (`/montecarlo`) | ✅ **Live** | Seeded GBM, 10,000 paths, μ/σ from the ticker's real returns; VaR, CVaR, target probability |
| Explainability ("لماذا هذه التوصية") | ✅ **Live** | Client-side decomposition of the existing composite into pillar contributions |
| Position sizing (`/position-size`) | ✅ **Live** | Volatility targeting + ¼ Kelly (capped) + 2×ATR stop; formulas shown on demand. p_win mapping from composite is calibration-beta |
| Risk profiling (3-question KYC-lite) | ✅ **Live** | محافظ / متوازن / جريء — adjusts sizing, thresholds, and optimizer live |
| Shariah screening (`/shariah`) | 🟡 **Beta — تجريبي / قيد المعايرة** | Real AAOIFI ratio math (debt/cash/receivables ÷ mcap ≤ 30%) from Yahoo balance sheets + business-activity filter. **TODO(data):** non-compliant-income % and purification rate need a segment-revenue feed (IdealRatings/Refinitiv) — returned as `null`, never faked |
| Halal portfolio optimizer (`/optimize`) | 🟡 **Beta — demo basket** | Real Markowitz (no-short) on 1y returns of a clearly-labeled demo basket of US stocks held by Shariah ETFs (SPUS/HLAL). **TODO(data):** wire basket to the live Shariah screen over a Tadawul+US universe |
| Backtest + calibration (`/backtest`) | 🟡 **Partial** | Real 2y walk-forward of the **technical pillar** (hit rate, Brier, Sharpe, max DD, equity curve). Full 4-pillar validation requires historical sentiment/analyst archives — labeled **«هدف التحقق»** (validation target), not achieved |

Saudi tickers are supported through Yahoo Finance's `.SR` suffix (e.g. `2222.SR` for Aramco) — prices/history are solid, but news and analyst coverage can be thin; the copilot says so when it happens.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/analyze?ticker=AAPL` | Full pipeline — SG denoising, Z-Score, FinBERT, inference weights, SMAs, RSI, MACD, BB, OBV, Volume, Analysts, Price Target, F&G, News, GPT-4o (cached 5 min) |
| `POST` | `/chat` | Arabic copilot — GPT-4o tool-calling over analyze / montecarlo / shariah / position-size / backtest. Body: `{"message": "أشتري أرامكو؟", "history": []}` |
| `GET` | `/montecarlo?ticker=AAPL&horizon_days=63&target=250` | Seeded GBM, 10k paths: percentile cone (P05–P95), sample paths, VaR/CVaR 95%, P(reach target). Deterministic per ticker+day |
| `GET` | `/position-size?ticker=AAPL&profile=balanced&score=71&capital=100000` | % of capital (vol targeting + capped ¼ Kelly), 2×ATR stop, formulas |
| `GET` | `/shariah?ticker=AAPL` | AAOIFI-style screen → حلال / مختلط / غير متوافق (beta) |
| `GET` | `/optimize?risk_profile=balanced` | Efficient frontier + optimal allocation on halal demo basket (beta) |
| `GET` | `/backtest?ticker=AAPL&threshold=55` | 2y walk-forward technical-pillar signal vs buy & hold |
| `GET` | `/health` · `/sma` · `/fear-greed` · `/news` · `DELETE /cache` | Utilities (unchanged) |

Interactive docs at `/docs`.

## Frontend Pages

| Page | Language | What it is |
|------|----------|-----------|
| `index.html` | العربية (RTL) | Clean product landing — live API health, module grid with honest live/beta tags |
| `terminal.html` | العربية (RTL) | **Flagship quant terminal** — copilot chat, composite gauge + score attribution bars, Shariah badge, position sizing, Monte Carlo fan chart, walk-forward backtest, halal frontier, 9-module strip, FinBERT news |
| `dashboard.html` | English | Classic terminal (unchanged, still fully functional) |
| `compare.html` | English | Side-by-side two-ticker comparison (unchanged) |

Frontend niceties:
- `terminal.html?ticker=AAPL` deep-links an analysis.
- `terminal.html?record=1` hides nav/footer chrome for clean 16:9 screen recording.
- `?api=http://localhost:7860` points any page at a local backend.
- Risk profile persists in `localStorage` and re-calibrates position sizing, signal thresholds, and the optimizer live.

## Architecture

```
User (Arabic) ──► terminal.html ──► GET /analyze ─┬─ yfinance OHLCV/news/analysts/targets
       │                                          ├─ CNN Fear & Greed
       │                                          ├─ FinBERT (HF Inference API)
       │                                          └─ GPT-4o hardline-quant verdict
       │
       ├──► POST /chat ──► GPT-4o tool-calling ──► analyze / montecarlo / shariah /
       │                   (numbers only from tools)  position-size / backtest
       ├──► GET /montecarlo  (seeded GBM 10k paths — μ,σ from real returns)
       ├──► GET /position-size (vol targeting + ¼ Kelly + 2×ATR stop)
       ├──► GET /shariah     (AAOIFI ratios — beta, income feed pending)
       ├──► GET /optimize    (Markowitz no-short on halal demo basket — beta)
       └──► GET /backtest    (2y walk-forward technical pillar)
```

All fetches run concurrently (`asyncio.gather`); OHLCV history is shared across endpoints via a 5-minute TTL cache.

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, FastAPI, Uvicorn, Pydantic v2 |
| Data | yfinance (US + Tadawul `.SR`), CNN F&G, httpx, BeautifulSoup4 |
| Quant math | NumPy — Savitzky-Golay, Z-Score, GBM Monte Carlo, Markowitz sampling, walk-forward vectorized backtest, Kelly/ATR sizing |
| NLP | ProsusAI/finbert via HF Inference API |
| AI agent | OpenAI GPT-4o — strict JSON verdict + tool-calling copilot (temperature 0.2–0.3) |
| Caching | cachetools TTLCache (analysis + raw history, 5 min) |
| Frontend | Hand-rolled canvas charts (fan chart, equity curves, frontier), SVG gauges, IBM Plex Sans Arabic, RTL-first CSS — no chart libraries on the new pages |
| Deployment | Docker on Hugging Face Spaces (API) · Vercel / GitHub Pages (static frontend) |

## Run Locally

```bash
pip install -r requirements.txt
```

`.env`:
```
OPENAI_API_KEY=your_key_here     # required for GPT verdict + copilot
HF_API_TOKEN=your_hf_token_here  # optional: higher FinBERT rate limits
```

```bash
python app.py                    # API on http://localhost:7860 — docs at /docs
python -m http.server 8000       # serve the frontend
# open http://localhost:8000/terminal.html?api=http://localhost:7860
```

## Deploy to Hugging Face Spaces

1. Create a Space → SDK: **Docker**.
2. Push this repo (Dockerfile, app.py, requirements.txt are the backend).
3. Add `OPENAI_API_KEY` (and optionally `HF_API_TOKEN`) as Space Secrets.
4. The Space builds and serves on port **7860**. The static pages can be hosted anywhere (Vercel/GitHub Pages) — they point at the Space URL by default.

## Honesty Rules (enforced in code)

- The copilot may only state numbers that appear in tool outputs; tool failures are reported, not papered over.
- Monte Carlo is seeded per ticker+day → reproducible, and labeled as a probability distribution, not a forecast.
- Shariah screen and optimizer carry a permanent **تجريبي** badge until their data feeds are wired; missing figures return `null` and render as «قيد الربط», never as fake numbers.
- Backtest reports its true scope (technical pillar) and labels full-composite validation **«هدف التحقق»**.
- Nothing simulated is ever displayed as real.

## Demo

See [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md) for the 30-second Arabic hero-flow walkthrough (use `terminal.html?record=1`).
