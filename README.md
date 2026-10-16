# Smart Swarm

**Smart Swarm finds where your retail money leaks, then tells you which product to sell, in which region, at what price and discount, with every call backed by past sales and explained with SHAP.**

## What it does

- **Profit prediction** - pick a product, a discount, a quantity, a channel, a segment and a month; six trained models return the expected profit, a loss flag and the break-even discount.
- **Region plan** - `best` mode ranks all five regions and names one; `options` mode takes a region and returns a discount ladder (0-25%) with the maximum safe discount and a suggested price.
- **Historical evidence** - every prediction is shown next to what actually happened for that product, its category and each region, including the inventory row and similar past sales.
- **SHAP explanations** - each answer carries a base value plus per-feature contributions that add back up to the prediction, so the number is auditable, not a black box.
- **Ask box** - plain-English questions ("which products lose the most money?") are answered by SQL retrieval over the SQLite database, and the rows used are shown.

## Screenshot

Add a screenshot at `docs/screenshot.png` (the file in this repo is a labelled placeholder) and it will render here.

## Tech stack

| Layer | Tool |
|---|---|
| API | FastAPI |
| Database | SQLite |
| Data wrangling | pandas |
| Models | scikit-learn, XGBoost |
| Explanations | SHAP |
| Frontend | React + TypeScript + Vite + Tailwind |

## Folder structure

```
smart-swarm/
├── backend/            FastAPI app, config, database build and the query/explain layer
│   ├── main.py         routes: /health, /api/*, plus /charts and the built frontend at /
│   ├── config.py       finds the five CSVs, declares the feature columns and output paths
│   ├── build_db.py     cleans the CSVs and writes data/db/money_leak.db
│   └── rag.py          SQL queries, profit and region prediction, SHAP, the Ask box
├── ml_model/           training code and its artefacts
│   ├── train.py        the pipeline, runnable as a module
│   ├── train.ipynb     the same pipeline as a Kaggle-style notebook
│   └── models/         bundle.joblib: the six models plus the category encoder maps
├── data/
│   ├── raw/            the five source CSVs, opened read-only and never modified
│   └── db/             money_leak.db, rebuilt in under a second
├── evidence/
│   ├── metrics.json    model registry, self-checks and every metric the UI shows
│   ├── charts/         18 PNG charts, opened full size in the Model Evidence tab
│   └── logs/           the last training log
├── frontend/           React + TypeScript + Vite + Tailwind app
│   ├── src/            components (Dashboard, Predict, Ask, Evidence) and lib/api.ts
│   └── dist/           production build served by FastAPI at /
├── docs/               screenshot and extras
├── requirements.txt    pinned Python dependencies
└── README.md
```

## Prerequisites

- **Python 3.11 or newer** - developed and verified on **Python 3.14.3**, which is the only version this was tested with
- **Node 20 or newer and npm** - verified on **Node v24.18.0 / npm 11.16.0** (Tailwind 4 and Vite 5 need Node 20+)
- **The five source CSVs** in the project folder, or in `./datasets/`, or in `./data/`. They are only ever read, never generated:

| File | Rows |
|---|---|
| customers.csv | 3000 |
| products.csv | 250 |
| transactions.csv | 18000 |
| inventory.csv | 5000 |
| campaigns.csv | 1200 |

`backend/config.py` searches those folders, prints the folder it chose and stops with a named error if a file is missing or its row count does not match. Here it finds `smart-swarm/data/raw/`.

## How to run

```bash
# 1. Python environment (Windows)
python -m venv .venv
.venv\Scripts\activate
# macOS / Linux
python -m venv .venv && source .venv/bin/activate

# 2. Dependencies
pip install -r requirements.txt

# 3. Database (measured: ~0.9 s)
python -m backend.build_db

# 4. Train (measured: ~6.7 s)
python -m ml_model.train

# 5. Frontend (measured: ~2 s after npm install)
cd frontend
npm install
npm run build
cd ..

# 6. Serve the API and the built app (measured: 2.2 s to a healthy /health)
uvicorn backend.main:app --port 8000
# then open http://localhost:8000
```

Each step writes something you can inspect: step 3 rewrites `data/db/money_leak.db` (10 tables) and prints the cleaning counts, step 4 writes `ml_model/models/bundle.joblib`, `evidence/metrics.json` and the 18 charts in `evidence/charts/`, step 5 writes `frontend/dist/`.

**Optional dev mode** - two terminals: `uvicorn backend.main:app --port 8000 --reload` for the API, and `cd frontend && npm run dev` for Vite with hot reload. `vite.config.ts` proxies `/api` and `/charts` to port 8000, so no CORS setup is needed.

**Run the training as a notebook** - open `ml_model/train.ipynb` in Jupyter or upload it to Kaggle. The first cell walks up to the project root and adds it to `sys.path`, the next cells are the pipeline step by step, then `main()` runs once, prints the evidence and shows the charts inline. The notebook code is line-for-line the same as `ml_model/train.py`, so both paths produce identical artefacts.

## API reference

| Method | Path | Purpose | Example request body |
|---|---|---|---|
| GET | `/health` | bundle and SHAP explainer status, database path | - |
| GET | `/api/options` | products, categories, channels, segments, months, regions | - |
| GET | `/api/summary` | revenue, lost revenue share, stale pending orders, worst products, margins | - |
| GET | `/api/products?category=&limit=` | products ranked by completed profit, worst first | - |
| GET | `/api/product/{product_id}?channel=&customer_segment=` | one product with its history and latest inventory row | - |
| POST | `/api/predict` | profit from three models, loss flag, break-even discount, SHAP, reasoning | `{"product_id":"PR0001","discount_pct":0.1,"quantity":3,"channel":"Web","customer_segment":"Loyal","month":11}` |
| POST | `/api/region-plan` | `region: null` ranks all five regions; a region name returns the discount ladder | same body plus `"region":"North"` |
| POST | `/api/ask` | SQL retrieval for a question about worst products, stale orders, stockouts or margins | `{"question":"which products lose the most money?"}` |
| GET | `/api/category-cards` | one card per category: margin, share below cost, campaign CPA | - |
| GET | `/api/campaign-cards` | one card per channel: spend, median CPA and CVR | - |
| GET | `/api/metrics` | the model registry, the self-checks and every metric in `evidence/metrics.json` | - |
| GET | `/charts/{name}.png` | the saved evidence charts | - |
| GET | `/` | the built React app from `frontend/dist` | - |

Worked example on the live server: `POST /api/predict` for `PR0001`, 10% discount, quantity 3 returns XGB profit **216.3907**, `loss_flag false`, break-even discount **0.7703**; `POST /api/region-plan` with `region: null` returns **5 ranked regions** (recommended Central, with the tie banner) and `region: "North"` returns a **6-row ladder** from 0% to 25%, max safe discount **0.7447**, suggested price **79.33**.

## Models

| Model | Task | Dataset | Features | Target | Metric | Real value |
|---|---|---|---|---|---|---|
| LinearRegression | profit | Dataset 1 (one-hot) | 9 | profit | R2 / MAE | R2 0.8253, MAE 129.12 |
| DecisionTreeRegressor | profit | Dataset 2 (encoded) | 9 | profit | R2 / MAE | R2 0.9816, MAE 39.46 |
| XGBRegressor | profit | Dataset 2 (encoded) | 9 | profit | R2 / MAE | **R2 0.9994, MAE 8.20** |
| LogisticRegression | winning region | region_winners | 4 | winner | accuracy / log-loss | acc 0.1814, log-loss 1.6402 |
| DecisionTreeClassifier | winning region | region_winners | 4 | winner | accuracy / log-loss | acc 0.1953, log-loss 1.6585 |
| XGBClassifier | winning region | region_winners | 4 | winner | accuracy / log-loss | **acc 0.1953, log-loss 1.7773** |

The last column of each row is the production model. The baseline MAE for profit is **324.51** (predict the mean), so all three regressors beat it. Region accuracy has to be read against **chance 0.20** and the **majority baseline 0.2279**.

**Held-out split:** months **1-9 train** (13,581 rows), months **10-12 test** (4,419 rows). Both XGBoost models use `eval_set` on the test months, and SHAP reconciles to the prediction within **0.001099** across 1,500 sampled rows (0.000028 on the live `/api/predict` call above).

Not trained, on purpose: returns, cancellations and stockouts. See the honest findings below.

## Data cleaning

`backend/build_db.py` prints these counts on every run:

- **76 negative-margin products** flagged (`unit_cost >= list_price`): they can never be sold at a profit, so they are excluded from "sell more" advice.
- **797 pre-signup sales** flagged: a transaction timestamped before the customer signed up is a data-integrity leak, kept but marked.
- **824 of 918 pending orders are stale** - pending long enough that they will not complete; they are excluded from profit and shown as stuck revenue instead.
- **6,143 of 18,000 sales (34.1%) lose money**, and cancelled or returned orders are **17.2% of revenue**. Both are reported as leaks, not hidden.
- **84 campaign rows have clicks above impressions** and **88 have conversions above clicks**; those rows are dropped from the campaign cards.
- **10 true stockouts** and **717 at-risk inventory rows** are separated, so "out of stock" is not confused with "stock arrives late".
- Revenue is recomputed as `quantity * unit_price` (the price is already net of discount) and profit as `revenue - quantity * unit_cost`; discount is never applied twice.
- `region_winners` holds **863** category-month cells, and its winner shares (North 23.3%, South 19.6%, West 19.6%, Central 19.5%, East 18.1%) are the only input to the region classifiers.

## Honest findings

- **Profit is mostly arithmetic of price, cost and discount**, so a near-perfect R2 is expected, not impressive. The model is a *what-if guardrail* (does this discount still leave a profit, and where is break-even), not a demand forecast.
- **Regions are statistically tied.** Kruskal-Wallis on profit across regions gives **p = 0.9549**, and no category reaches significance (Beauty 0.4994, Electronics 0.7024, Fashion 0.1063, Grocery 0.9345, Home 0.4758, Sports 0.0999). Profit per completed sale is Central 186.32, East 184.15, North 178.08, South 183.85, West 190.14 - close enough that the ranking is indicative only, which is why the app prints the tie banner and never claims a clear winner. The region classifier confirms it: best test accuracy 0.1953 against 0.20 chance and 0.2279 majority, log-loss above the uniform 1.6094, ROC-AUC ~0.5.
- **Campaigns cannot be linked to sales.** There is no campaign key on transactions, so spend, CPA and CVR are shown per channel on their own and never mixed into profit.
- **Returns, cancellations and stockouts were not modelled** because the data carries no signal for them; the counts above are reported instead of a fake model.

## Troubleshooting

| Problem | Fix |
|---|---|
| `STOP: the five raw CSVs are not present` | Put `customers.csv`, `products.csv`, `transactions.csv`, `inventory.csv`, `campaigns.csv` in `smart-swarm/data/raw/`, or in `smart-swarm/`, `smart-swarm/datasets/` or `smart-swarm/data/`. The error names the folders searched. |
| Row-count mismatch at startup | The app expects 3000 / 250 / 18000 / 5000 / 1200 rows. The message prints expected vs actual per file. |
| Port 8000 already in use | `uvicorn backend.main:app --port 8001`, or stop the old process. On Windows: `netstat -ano \| findstr :8000` then `taskkill /PID <pid> /F`. |
| `/api/*` answers `{"error": "Model bundle not loaded"}` | Train first: `python -m ml_model.train`. The 503 response repeats the fix. |
| `GET /` returns nothing / "FRONTEND NOT BUILT" in the log | `cd frontend && npm install && npm run build`. The API keeps working without it; only the UI is missing. |
| `npm run build` fails on TypeScript | The build runs `tsc -b` first. Fix the reported type error, or run `npx vite build` to emit the bundle anyway. |
| `npm: command not found` / old Node | Install Node 20+ from nodejs.org; verify with `node -v` and `npm -v`. |
| `pip install shap` or `xgboost` fails | Both ship wheels for CPython 3.11-3.14 on Windows, macOS and Linux. On an unusual platform, upgrade `pip` first: `python -m pip install -U pip`. |
| Dashboard loads but charts are blank | The images come from `/charts`; check `evidence/charts/` has 18 PNGs (run `python -m ml_model.train`). |

## Team

Team Instantiated Unit by S. Ariyan Minaxi and Tanush Kumar.
