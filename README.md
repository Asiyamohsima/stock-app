# StockSense — Stock Market Prediction Web App

A Flask app that forecasts a stock's next 7 trading days using ARIMA, an LSTM
neural network, and Linear Regression — trained fresh on that ticker's own
2-year history every time you search.

## What was fixed from the original project

1. **Every stock showed the same chart.** The old templates used static
   image files instead of generating a chart from the actual data returned
   for each symbol. Charts are now built with Matplotlib on every request,
   from that specific ticker's own price history, and embedded directly in
   the page (no image caching issues, no shared files).
2. **Dashboard / About / Contact all rendered the same page.** All three
   routes pointed at `index.html`. Each now has its own template
   (`dashboard.html`, `about.html`, `contact.html`) with real, distinct
   content — a live snapshot dashboard, an explanation of the models, and a
   working contact form (front-end only; wire up an email backend to send
   real messages).
3. **Predictions were single numbers, not a real forecast.** The original
   `ARIMA_ALGO` / `LSTM_ALGO` / `LIN_REG_ALGO` each returned one value (and
   the "LSTM" was just a 10-day average, not a neural network at all). They
   now produce genuine 7-day-ahead forecasts, shown both as a chart and as a
   day-by-day table. A real Keras LSTM is used when TensorFlow is installed;
   otherwise the app falls back to exponential smoothing rather than
   silently repeating one number.
4. **Hardcoded API keys.** The old `constants.py` committed real-looking
   Twitter/X API keys directly into source control. Since the sentiment
   feature was already stubbed out (disabled) in the code you provided, it's
   been removed rather than re-enabled with more embedded secrets. If you
   want to re-add tweet-based sentiment, keep the keys in a `.env` file
   (never commit it) and load them with `python-dotenv`.
5. **Broken dependency.** `requirements.txt` pointed at a dead
   `tweet-preprocessor` tarball URL that would fail `pip install`. Removed,
   along with other unused packages (streamlit, seaborn, alpha_vantage,
   nltk, textblob, tweepy, gunicorn) that had nothing to do with this Flask
   app's actual code path.

## Project structure

```
main.py                  Flask app: routes, data fetch, models, charts
templates/
  base.html               Shared layout + navigation
  index.html               Home page with the ticker search form
  dashboard.html            Live snapshot cards for a few tickers
  about.html                 Explanation of the three models
  contact.html                 Contact form
  results.html                 Forecast charts + table for a searched ticker
static/
  css/style.css             All styling
  js/script.js              Small UI helper
requirements.txt
```

## Running it

```bash
pip install -r requirements.txt
python main.py
```

Then open http://127.0.0.1:5000/

### Making the Contact Us form actually send email

The Contact page sends real email via SMTP. Copy `.env.example` to `.env`
and fill in your details (or export the same variables in your shell):

```bash
cp .env.example .env
```

For Gmail:
1. Turn on 2-Step Verification on the sending Google account.
2. Create an App Password at https://myaccount.google.com/apppasswords.
3. Put that 16-character password in `SMTP_PASSWORD` (not your normal Gmail
   password).

Any other SMTP provider works too — just change `SMTP_SERVER` / `SMTP_PORT`.
If these variables aren't set, the form still works but shows a friendly
"couldn't be sent" message instead of crashing.

- **TensorFlow is optional.** If it's not installed, the LSTM forecast
  automatically falls back to exponential smoothing instead of crashing.
- Indian tickers can be entered without the exchange suffix (`TCS`, `INFY`,
  `RELIANCE`, etc.) — they're mapped to the NSE (`.NS`) automatically.
  Everything else is treated as a NASDAQ/US ticker.

## Disclaimer

This is an educational demo. Nothing here is financial advice, and no model
can reliably predict real market prices.
