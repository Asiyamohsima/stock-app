# StockSense — Full Project Documentation

## 1. What this app does

StockSense is a web app where a user types in a stock ticker (e.g. `AAPL`,
`TSLA`, `TCS`, `INFY`) and instantly gets:

- The stock's **current price** and **2-year average price**.
- A **7-day-ahead price forecast**, generated primarily by an **LSTM
  (Long Short-Term Memory) deep learning model**, with ARIMA and Linear
  Regression forecasts shown alongside it for comparison.
- Two **charts**: recent price history, and the 7-day forecast plotted
  against the last 45 days of real data.
- A simple **BUY / SELL / HOLD** recommendation based on whether the
  forecast average is above or below today's price.
- A **live dashboard** of a handful of popular tickers.
- An **About** page explaining the app, and a **Contact Us** page that
  sends real email.

It supports both **NASDAQ** (US) tickers and **NSE** (Indian) tickers —
Indian symbols like `TCS`, `INFY`, `RELIANCE` are automatically mapped to
their Yahoo Finance `.NS` suffix behind the scenes, so the user never has to
know that detail.

---

## 2. Tech stack at a glance

| Layer | Technology |
|---|---|
| Backend framework | **Flask** (Python) |
| Data source | **yfinance** (pulls live data from Yahoo Finance) |
| Forecasting models | **LSTM** (TensorFlow/Keras), **ARIMA** (statsmodels), **Linear Regression** (scikit-learn) |
| Data handling | **pandas**, **NumPy** |
| Charting | **Matplotlib**, rendered server-side and embedded as base64 images |
| Frontend | Server-rendered **HTML** via Jinja2 templates, plain **CSS**, a little vanilla **JavaScript** |
| Email | Python's built-in **smtplib** (SMTP) |
| Config | Environment variables, optionally loaded from a `.env` file via **python-dotenv** |

There is no separate frontend framework (no React/Vue) and no database —
everything is computed fresh on each request and rendered straight to HTML.
This keeps the project simple and easy to run/grade for a college project.

---

## 3. Project structure

```
stock-app/
├── main.py                 All backend logic: routes, data fetching,
│                            models, chart generation, email
├── requirements.txt         Python dependencies
├── .env.example              Template for email (SMTP) configuration
├── README.md                  Setup instructions
├── templates/                  Jinja2 HTML templates (the "views")
│   ├── base.html                Shared layout: navbar + footer
│   ├── index.html                 Home page (ticker search form)
│   ├── dashboard.html               Live snapshot cards
│   ├── about.html                     About StockSense
│   ├── contact.html                     Contact form
│   └── results.html                       Forecast results page
└── static/
    ├── css/style.css            All styling
    └── js/script.js               Small UI helper (quick-fill chips)
```

---

## 4. Backend — how it's implemented (`main.py`)

The backend is a single Flask application. Below is what each part does,
roughly in the order a request flows through it.

### 4.1 App setup

- `app = Flask(__name__)` creates the app.
- `app.secret_key` is required for **flash messages** (the little success/
  error banners on the Contact page) to work securely.
- `@app.context_processor` injects `current_year` into every template, so
  the footer can show `© 2026 StockSense` without repeating that logic on
  every page.
- `@app.after_request` adds `Cache-Control: no-store` headers to every
  response, so the browser never shows a stale cached page — important
  because prices and forecasts change constantly.
- TensorFlow's Keras `LSTM`/`Dense`/`Sequential` classes are imported
  inside a `try/except`. If TensorFlow isn't installed, `KERAS_AVAILABLE`
  is set to `False` and the app automatically falls back to a simpler
  forecasting method (see §4.4) instead of crashing.

### 4.2 Routes (the pages/URLs)

| Route | Method | Purpose |
|---|---|---|
| `/` | GET | Home page with the ticker search form |
| `/dashboard` | GET | Live snapshot cards for popular tickers |
| `/know-more/`, `/about` | GET | About page |
| `/contact` | GET, POST | Contact form (GET shows it, POST sends the email) |
| `/insertintotable` | POST | Runs the forecast for a submitted ticker and shows `results.html` |

Each route renders its **own** template with an `active` flag (e.g.
`active="dashboard"`), which the navbar uses to highlight the current page —
this is what fixes the earlier bug where every page looked identical.

### 4.3 Fetching stock data — `get_historical()`

1. `resolve_ticker()` cleans up the symbol: uppercases it, corrects the
   common typo `APPL` → `AAPL`, and appends `.NS` if the symbol is in a
   hardcoded set of well-known Indian stocks (`INDIAN_SYMBOLS`).
2. `yf.download(ticker, period="2y", ...)` pulls **2 years of daily OHLCV**
   (Open/High/Low/Close/Volume) data from Yahoo Finance.
3. The function then cleans the data:
   - Flattens Yahoo's sometimes-multi-level column headers.
   - Makes sure there's a `Date` column (renaming `Datetime` if needed).
   - Confirms all required columns exist; bails out (`return None`) if any
     are missing.
   - Converts price/volume columns to numeric, dropping rows where `Close`
     is missing.
4. Returns a clean `pandas.DataFrame` plus the "clean" symbol name (without
   the `.NS` suffix, so the UI can show `TCS` instead of `TCS.NS`).

If anything goes wrong at any step (bad ticker, network hiccup, Yahoo
returning nothing), the function returns `None`, and the route shows the
home page again with a friendly "couldn't find data" message instead of
crashing.

### 4.4 The three forecasting models

All three take the stock's `Close` price series and each return a list of
**7 predicted prices** (one per upcoming trading day) plus an optional
"note" string explaining any fallback that was used.

**LSTM (`LSTM_ALGO`) — the primary model**
- Scales closing prices to a 0–1 range with `MinMaxScaler` (neural networks
  train better on normalized data).
- Builds sliding windows of the last 10 days to predict the next day
  (`_build_windows`).
- Trains a small `Sequential` Keras model: one `LSTM(32)` layer followed by
  a `Dense(1)` output layer, for 8 epochs.
- To forecast 7 days ahead, it predicts one day, appends that prediction to
  the input sequence, and repeats — a technique called **iterative /
  recursive multi-step forecasting**.
- Un-scales the predictions back to real price values.
- **Fallback:** if TensorFlow isn't installed, or there isn't enough
  history (less than 30 data points), it uses **exponential smoothing**
  plus a simple trend term instead — still a real, per-stock calculation,
  not a fixed/fake number.

**ARIMA (`ARIMA_ALGO`)**
- A classical statistical time-series model (AutoRegressive Integrated
  Moving Average) from `statsmodels`, configured with order `(5, 1, 0)`.
- Captures autocorrelation and short-term trend in the price series.
- `.forecast(steps=7)` produces the 7-day-ahead prediction directly.

**Linear Regression (`LIN_REG_ALGO`)**
- Fits a straight line through the *entire* 2-year closing-price history
  (`X` = day index, `y` = price) using scikit-learn.
- Extends that line 7 days past the last known day.
- Acts as a simple baseline to compare the other two models against.

Every model function is wrapped in a `try/except` — if a model fails for
any reason, it falls back to repeating the last known price rather than
breaking the whole page.

### 4.5 Turning forecasts into a recommendation

`recommending(today_price, forecast_avg)` averages all 21 predicted values
(7 days × 3 models) and compares that average to today's price:

- Average **above** today's price → **BUY**
- Average **below** today's price → **SELL**
- Roughly equal → **HOLD**

This is intentionally simple and clearly labeled as educational — it's not
real investment advice.

### 4.6 Chart generation

Charts are built with Matplotlib **on every request**, directly from that
specific stock's own data, then converted into a **base64-encoded PNG** and
embedded straight into the HTML (`fig_to_base64`). This means:

- No image files are saved to disk or reused between users/stocks.
- Every ticker you search produces a genuinely different chart.

Two charts are produced:
1. **`make_history_chart`** — the last 180 trading days of closing prices,
   as a filled line chart.
2. **`make_forecast_chart`** — the last 45 days of real data, plus all
   three models' 7-day forecasts plotted as dashed lines past a vertical
   "today" marker, so you can visually compare how the models diverge.

### 4.7 The `/insertintotable` route (main prediction flow)

This is the route the search form submits to. Step by step:

1. Reads the `nm` form field (the ticker the user typed).
2. Calls `get_historical()`. If it fails, re-shows the home page with an
   error.
3. Computes `today_price` and `mean_price` from the data.
4. Runs all three forecasting models.
5. Computes the average forecast and the BUY/SELL/HOLD recommendation.
6. Builds both charts.
7. Builds a day-by-day forecast table (date + each model's predicted
   price).
8. Renders `results.html` with everything above.

### 4.8 Contact form email (`send_contact_email`)

- Reads SMTP credentials from environment variables (`SMTP_SERVER`,
  `SMTP_PORT`, `SMTP_USERNAME`, `SMTP_PASSWORD`, `CONTACT_RECEIVER_EMAIL`),
  which can be set directly or loaded automatically from a local `.env`
  file via `python-dotenv`.
- Builds a plain-text email (`email.mime.text.MIMEText`) with the visitor's
  name, email, and message, and sets `Reply-To` to the visitor's email so
  you can reply directly from your inbox.
- Connects to the SMTP server with `smtplib.SMTP(...)`, starts TLS
  encryption, logs in, and sends the message.
- If credentials aren't configured, or sending fails for any reason, the
  form shows a friendly error via Flask's **flash messages** instead of
  crashing — the visitor always gets a clear response either way.

### 4.9 Live dashboard (`/dashboard`)

For a fixed list of well-known tickers (`DASHBOARD_SYMBOLS`), it pulls the
last 5 days of data for each, works out the price change vs. the previous
day, and passes a list of small "card" dictionaries to the template. If
Yahoo Finance is unreachable for a given symbol, that symbol is silently
skipped rather than breaking the whole dashboard.

---

## 5. Frontend — how it's implemented

The frontend is **server-rendered** — Flask + Jinja2 build complete HTML
pages on the server, which are then styled with plain CSS. There's no
separate JavaScript framework or API layer between frontend and backend;
data goes straight from Python variables into the HTML template.

### 5.1 Templates (`templates/`)

- **`base.html`** is the shared layout every other page extends. It
  defines the navbar (Home / Dashboard / About / Contact Us, with the
  current page highlighted via the `active` variable) and the footer. Jinja2's
  `{% block content %}` is where each specific page injects its own body.
- **`index.html`** — the landing page: a hero section, the ticker search
  form (`<form action="/insertintotable" method="POST">`), quick-fill
  ticker "chips," and a 3-card feature summary.
- **`dashboard.html`** — loops over the `cards` list passed from Flask and
  renders one snapshot card per ticker, colored green/red depending on
  whether the price is up or down.
- **`about.html`** — static content explaining what the app does and why
  it's LSTM-based.
- **`contact.html`** — the contact form, plus `{% with messages =
  get_flashed_messages(...) %}` to display the success/error banner after
  a submission.
- **`results.html`** — the forecast results: stat cards (current price,
  average price, forecast average, recommendation), the two embedded
  chart images, and the day-by-day forecast table.

### 5.2 Styling (`static/css/style.css`)

Plain, hand-written CSS (no framework like Bootstrap/Tailwind). It defines:
- CSS custom properties (`--blue`, `--dark`, `--gray`, etc.) for a
  consistent color palette.
- A responsive grid layout using `grid-template-columns:
  repeat(auto-fit, minmax(...))` so dashboard cards, feature cards, etc.
  reflow nicely on smaller screens.
- Distinct styling per page section (hero, stat cards, chart blocks,
  dashboard cards, about cards, contact form).

### 5.3 JavaScript (`static/js/script.js`)

Very minimal — a single `fillSymbol(symbol)` function used by the "quick
pick" ticker chips on the home page, which just fills the search input with
a preset symbol (e.g. clicking "AAPL" fills the box with `AAPL`) so the
user can try the app with one click.

---

## 6. How a request flows end-to-end (example)

1. User opens `/` → Flask renders `index.html`.
2. User types `TCS` and clicks **Predict** → browser POSTs to
   `/insertintotable` with `nm=TCS`.
3. Flask's `insertintotable()`:
   - Calls `get_historical("TCS")` → recognizes it as an Indian stock →
     fetches `TCS.NS` from Yahoo Finance → returns a cleaned DataFrame.
   - Runs `ARIMA_ALGO`, `LSTM_ALGO`, `LIN_REG_ALGO` on the `Close` column.
   - Builds the recommendation and both charts.
4. Flask renders `results.html` with all of that data, and the browser
   shows the price, recommendation, charts, and forecast table for **TCS
   specifically** — every number and every chart pixel comes from TCS's own
   data.

---

## 7. Configuration & environment variables

| Variable | Purpose | Required? |
|---|---|---|
| `FLASK_SECRET_KEY` | Signs session/flash-message cookies | Recommended |
| `SMTP_SERVER` | Outgoing mail server (default `smtp.gmail.com`) | For Contact Us |
| `SMTP_PORT` | Mail server port (default `587`) | For Contact Us |
| `SMTP_USERNAME` | The sending email address | For Contact Us |
| `SMTP_PASSWORD` | App password / SMTP password | For Contact Us |
| `CONTACT_RECEIVER_EMAIL` | Where contact messages are delivered | For Contact Us |

These can be set as real OS environment variables, or placed in a `.env`
file (copy `.env.example`) which is auto-loaded on startup.

---

## 8. Limitations & honest caveats

- **Not financial advice.** The BUY/SELL/HOLD logic is a simple average
  comparison, not real trading logic.
- **LSTM training happens on every request**, using only 8 epochs on a
  small window — this favors speed over prediction accuracy, which is
  reasonable for a demo/class project but not production-grade forecasting.
- **No database** — nothing is persisted between requests; every prediction
  is computed fresh.
- **No user accounts/authentication** — anyone who can reach the app can
  use every feature.
- Relies on **Yahoo Finance being reachable** at request time; if it's
  down or rate-limits a request, that specific lookup will fail gracefully
  with an on-page message.

---

## 9. Possible future improvements

- Cache recent lookups (e.g. with Flask-Caching) so repeated searches for
  the same ticker within a few minutes don't re-fetch/re-train.
- Persist model accuracy metrics (e.g. backtested error) so predictions
  come with a confidence indicator.
- Add user accounts to save a personal watchlist.
- Swap the hand-rolled SMTP contact form for a transactional email API
  (SendGrid, Mailgun, etc.) for better deliverability at scale.
