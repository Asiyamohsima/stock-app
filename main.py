"""
Stock Market Prediction Web App
--------------------------------
Flask backend that pulls real historical data per symbol (Yahoo Finance),
trains ARIMA / LSTM / Linear-Regression models on THAT symbol's own data,
and renders a unique chart for every request (no more static/shared images).
"""

import base64
import io
import os
import smtplib
import warnings
from datetime import datetime, timedelta
from email.mime.text import MIMEText

import matplotlib

matplotlib.use("Agg")  # headless rendering - required on servers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, flash, redirect, render_template, request, url_for
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")

# Optional: auto-load a local .env file if python-dotenv is installed.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass

# Optional heavy dependency - app must still run without it.
try:
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.models import Sequential

    KERAS_AVAILABLE = True
except Exception:
    KERAS_AVAILABLE = False

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-key-change-me")


@app.context_processor
def inject_year():
    return {"current_year": datetime.now().year}
plt.style.use("ggplot")

FORECAST_DAYS = 7

# -------------------- Email (Contact Us) config --------------------
# Set these as real environment variables before running the app.
# For Gmail: use an "App Password" (not your normal password) - see README.
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME")  # the sending mailbox
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")  # app password / SMTP password
CONTACT_RECEIVER_EMAIL = os.environ.get("CONTACT_RECEIVER_EMAIL", SMTP_USERNAME)

INDIAN_SYMBOLS = {
    "RELIANCE", "TCS", "INFY", "HDFCBANK", "ICICIBANK", "SBIN", "ITC",
    "WIPRO", "HCLTECH", "BHARTIARTL", "AXISBANK", "KOTAKBANK", "LT",
    "MARUTI", "TATAMOTORS", "TATASTEEL", "ADANIENT", "ADANIPORTS",
    "SUNPHARMA", "ONGC",
}

DASHBOARD_SYMBOLS = ["AAPL", "MSFT", "GOOGL", "TSLA", "INFY.NS", "TCS.NS"]


# =========================================================
# NO CACHE (every page/asset always freshly rendered)
# =========================================================
@app.after_request
def add_header(response):
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


# =========================================================
# STATIC PAGES  (each route now has its OWN template/content)
# =========================================================
@app.route("/")
def index():
    return render_template("index.html", active="home")


@app.route("/dashboard")
def dashboard():
    """Live snapshot cards for a handful of well-known tickers."""
    cards = []
    for sym in DASHBOARD_SYMBOLS:
        try:
            hist = yf.download(sym, period="5d", progress=False, auto_adjust=False)
            if hist is None or hist.empty:
                continue
            if isinstance(hist.columns, pd.MultiIndex):
                hist.columns = hist.columns.get_level_values(0)
            last_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2]) if len(hist) > 1 else last_close
            change = last_close - prev_close
            pct = (change / prev_close * 100) if prev_close else 0
            cards.append({
                "symbol": sym.replace(".NS", ""),
                "price": round(last_close, 2),
                "change": round(change, 2),
                "pct": round(pct, 2),
                "up": change >= 0,
            })
        except Exception as e:
            print("Dashboard fetch error for", sym, ":", e)
    return render_template("dashboard.html", active="dashboard", cards=cards)


@app.route("/know-more/")
@app.route("/about")
def know_more():
    return render_template("about.html", active="about")


@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        message = request.form.get("message", "").strip()

        if not (name and email and message):
            flash("Please fill in every field before sending.", "error")
            return redirect(url_for("contact"))

        ok, error = send_contact_email(name, email, message)
        if ok:
            flash("Thanks! Your message has been sent - we'll get back to you soon.", "success")
        else:
            print("Contact email failed:", error)
            flash(
                "Sorry, your message couldn't be sent right now (email isn't configured "
                "on this server yet). Please try again later.",
                "error",
            )
        return redirect(url_for("contact"))

    return render_template("contact.html", active="contact")


def send_contact_email(name, sender_email, message):
    """Send the contact-form message via SMTP. Returns (success, error)."""
    if not (SMTP_USERNAME and SMTP_PASSWORD and CONTACT_RECEIVER_EMAIL):
        return False, "SMTP_USERNAME / SMTP_PASSWORD / CONTACT_RECEIVER_EMAIL not set"

    try:
        body = f"New message from StockSense contact form\n\nName: {name}\nEmail: {sender_email}\n\nMessage:\n{message}"
        msg = MIMEText(body)
        msg["Subject"] = f"StockSense contact form - {name}"
        msg["From"] = SMTP_USERNAME
        msg["To"] = CONTACT_RECEIVER_EMAIL
        msg["Reply-To"] = sender_email

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, [CONTACT_RECEIVER_EMAIL], msg.as_string())

        return True, None
    except Exception as e:
        return False, str(e)


# =========================================================
# DATA
# =========================================================
def resolve_ticker(symbol):
    symbol = symbol.strip().upper()
    if symbol == "APPL":
        symbol = "AAPL"
    if symbol in INDIAN_SYMBOLS:
        return symbol, symbol + ".NS"
    return symbol, symbol


def get_historical(symbol):
    try:
        clean_symbol, ticker = resolve_ticker(symbol)

        print(f"Fetching {ticker} ...")
        data = yf.download(ticker, period="2y", progress=False, auto_adjust=False)

        if data is None or data.empty:
            print("No data returned by Yahoo Finance for", ticker)
            return None, clean_symbol

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.loc[:, ~data.columns.duplicated()]

        data = data.reset_index()
        if "Date" not in data.columns:
            if "Datetime" in data.columns:
                data.rename(columns={"Datetime": "Date"}, inplace=True)
            else:
                return None, clean_symbol

        required = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required:
            if col not in data.columns:
                print("Missing column:", col)
                return None, clean_symbol

        if "Adj Close" not in data.columns:
            data["Adj Close"] = data["Close"]

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            data[col] = pd.to_numeric(data[col], errors="coerce")

        data.dropna(subset=["Close"], inplace=True)
        if data.empty:
            return None, clean_symbol

        return data, clean_symbol

    except Exception as e:
        print("ERROR IN get_historical():", e)
        return None, symbol


# =========================================================
# FORECASTING MODELS - each returns a 7-day-ahead list
# =========================================================
def ARIMA_ALGO(close_series):
    try:
        close_series = close_series.astype(float)
        if len(close_series) < 30:
            last = float(close_series.iloc[-1])
            return [round(last, 2)] * FORECAST_DAYS, "insufficient history"

        model = ARIMA(close_series, order=(5, 1, 0))
        fit = model.fit()
        forecast = fit.forecast(steps=FORECAST_DAYS)
        return [round(float(v), 2) for v in forecast], None
    except Exception as e:
        print("ARIMA error:", e)
        last = float(close_series.iloc[-1])
        return [round(last, 2)] * FORECAST_DAYS, str(e)


def _build_windows(values, window):
    X, y = [], []
    for i in range(window, len(values)):
        X.append(values[i - window:i])
        y.append(values[i])
    return np.array(X), np.array(y)


def LSTM_ALGO(close_series):
    """Real Keras LSTM when TensorFlow is installed, otherwise a robust
    exponential-smoothing fallback so the app still returns a sensible,
    symbol-specific forecast."""
    close_prices = close_series.astype(float).values
    window = 10

    if not KERAS_AVAILABLE or len(close_prices) < window * 3:
        # Exponential smoothing fallback (still per-symbol / data-driven).
        alpha = 0.3
        smoothed = close_prices[0]
        for p in close_prices[1:]:
            smoothed = alpha * p + (1 - alpha) * smoothed
        trend = (close_prices[-1] - close_prices[-window]) / window
        return [round(float(smoothed + trend * (i + 1)), 2) for i in range(FORECAST_DAYS)], \
            ("TensorFlow not installed - used exponential smoothing" if not KERAS_AVAILABLE else None)

    try:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(close_prices.reshape(-1, 1)).flatten()

        X, y = _build_windows(scaled, window)
        X = X.reshape((X.shape[0], X.shape[1], 1))

        model = Sequential([
            LSTM(32, activation="relu", input_shape=(window, 1)),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse")
        model.fit(X, y, epochs=8, batch_size=16, verbose=0)

        seq = scaled[-window:].tolist()
        preds_scaled = []
        for _ in range(FORECAST_DAYS):
            x_input = np.array(seq[-window:]).reshape((1, window, 1))
            next_val = model.predict(x_input, verbose=0)[0][0]
            preds_scaled.append(next_val)
            seq.append(next_val)

        preds = scaler.inverse_transform(np.array(preds_scaled).reshape(-1, 1)).flatten()
        return [round(float(v), 2) for v in preds], None
    except Exception as e:
        print("LSTM error:", e)
        last = float(close_prices[-1])
        return [round(last, 2)] * FORECAST_DAYS, str(e)


def LIN_REG_ALGO(close_series):
    try:
        close_prices = close_series.astype(float).values
        if len(close_prices) < 10:
            last = float(close_prices[-1])
            return [round(last, 2)] * FORECAST_DAYS, "insufficient history"

        X = np.arange(len(close_prices)).reshape(-1, 1)
        y = close_prices
        model = LinearRegression()
        model.fit(X, y)

        future_idx = np.arange(len(close_prices), len(close_prices) + FORECAST_DAYS).reshape(-1, 1)
        preds = model.predict(future_idx)
        return [round(float(v), 2) for v in preds], None
    except Exception as e:
        print("Linear Regression error:", e)
        last = float(close_series.iloc[-1])
        return [round(last, 2)] * FORECAST_DAYS, str(e)


def recommending(today_price, forecast_avg):
    try:
        if forecast_avg > today_price:
            return "BUY", "The average of next-7-day forecasts is above today's price."
        elif forecast_avg < today_price:
            return "SELL", "The average of next-7-day forecasts is below today's price."
        else:
            return "HOLD", "Forecasts are roughly flat versus today's price."
    except Exception:
        return "HOLD", "Not enough information to make a call."


# =========================================================
# CHART RENDERING  (base64 PNG, generated fresh per request/symbol)
# =========================================================
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_history_chart(df, symbol):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    plot_df = df.tail(180)
    ax.plot(plot_df["Date"], plot_df["Close"], color="#2563eb", linewidth=1.8)
    ax.fill_between(plot_df["Date"], plot_df["Close"], plot_df["Close"].min(), color="#2563eb", alpha=0.08)
    ax.set_title(f"{symbol} - Closing Price (last 180 trading days)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    fig.autofmt_xdate()
    return fig_to_base64(fig)


def make_forecast_chart(df, symbol, arima_fc, lstm_fc, lin_fc):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    recent = df.tail(45)
    ax.plot(recent["Date"], recent["Close"], color="#334155", linewidth=1.8, label="Historical Close")

    last_date = pd.to_datetime(df["Date"].iloc[-1])
    future_dates = [last_date + timedelta(days=i + 1) for i in range(FORECAST_DAYS)]

    ax.plot(future_dates, arima_fc, marker="o", linestyle="--", color="#dc2626", label="ARIMA forecast")
    ax.plot(future_dates, lstm_fc, marker="o", linestyle="--", color="#16a34a", label="LSTM forecast")
    ax.plot(future_dates, lin_fc, marker="o", linestyle="--", color="#f59e0b", label="Linear Regression forecast")

    ax.axvline(last_date, color="gray", linestyle=":", linewidth=1)
    ax.set_title(f"{symbol} - Next {FORECAST_DAYS}-Day Forecast")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend(loc="upper left", fontsize=8)
    fig.autofmt_xdate()
    return fig_to_base64(fig)


# =========================================================
# PREDICTION ROUTE
# =========================================================
@app.route("/insertintotable", methods=["POST"])
def insertintotable():
    symbol = request.form.get("nm", "").strip()

    if not symbol:
        return render_template("index.html", active="home", not_found=True)

    print("Stock entered:", symbol)

    df, clean_symbol = get_historical(symbol)

    if df is None or df.empty:
        return render_template("index.html", active="home", not_found=True,
                                searched_symbol=symbol)

    try:
        close_series = df["Close"]
        today_price = float(close_series.iloc[-1])
        mean_price = float(close_series.mean())

        arima_fc, arima_note = ARIMA_ALGO(close_series)
        lstm_fc, lstm_note = LSTM_ALGO(close_series)
        linear_fc, linear_note = LIN_REG_ALGO(close_series)

        forecast_avg = float(np.mean(arima_fc + lstm_fc + linear_fc))
        recommendation, reason = recommending(today_price, forecast_avg)

        history_chart = make_history_chart(df, clean_symbol)
        forecast_chart = make_forecast_chart(df, clean_symbol, arima_fc, lstm_fc, linear_fc)

        last_date = pd.to_datetime(df["Date"].iloc[-1])
        forecast_dates = [(last_date + timedelta(days=i + 1)).strftime("%a %d %b") for i in range(FORECAST_DAYS)]
        forecast_table = list(zip(forecast_dates, arima_fc, lstm_fc, linear_fc))

        return render_template(
            "results.html",
            active="home",
            quote=clean_symbol,
            today_price=round(today_price, 2),
            mean_price=round(mean_price, 2),
            forecast_avg=round(forecast_avg, 2),
            arima_pred=arima_fc[-1],
            lstm_pred=lstm_fc[-1],
            linear_pred=linear_fc[-1],
            arima_note=arima_note,
            lstm_note=lstm_note,
            linear_note=linear_note,
            forecast_table=forecast_table,
            history_chart=history_chart,
            forecast_chart=forecast_chart,
            recommendation=recommendation,
            reason=reason,
            last_updated=datetime.now().strftime("%d %b %Y, %I:%M %p"),
        )

    except Exception as e:
        print("ERROR WHILE CREATING RESULTS:", e)
        return render_template("index.html", active="home", not_found=True,
                                searched_symbol=symbol)


if __name__ == "__main__":
    app.run(debug=True)
