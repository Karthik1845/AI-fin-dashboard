import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.ensemble import IsolationForest
import numpy as np
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings("ignore")

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(
    page_title="🚀 AI Quant Financial Dashboard PRO",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------
# Auto Refresh (10 min)
# --------------------------------------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if datetime.now() - st.session_state.last_refresh > timedelta(minutes=10):
    st.session_state.clear()
    st.session_state.last_refresh = datetime.now()

# --------------------------------------------------
# Cached Data
# --------------------------------------------------
@st.cache_data(ttl=3600)
def load_stock_data(ticker, period):
    df = yf.download(ticker, period=period, progress=False)
    if df is None or df.empty:
        return None
    df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df.dropna()

@st.cache_data(ttl=3600)
def load_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "Market Cap": info.get("marketCap"),
        "P/E Ratio": info.get("trailingPE"),
        "Forward P/E": info.get("forwardPE"),
        "EPS": info.get("trailingEps"),
        "Dividend Yield": info.get("dividendYield"),
        "Beta": info.get("beta"),
        "52W High": info.get("fiftyTwoWeekHigh"),
        "52W Low": info.get("fiftyTwoWeekLow"),
        "Sector": info.get("sector"),
        "Industry": info.get("industry"),
    }

# --------------------------------------------------
# Indicators (UPGRADED)
# --------------------------------------------------
def indicators(df):
    df = df.copy()

    # Moving Averages
    df["SMA20"] = df["Close"].rolling(20).mean()
    df["SMA50"] = df["Close"].rolling(50).mean()
    df["EMA20"] = df["Close"].ewm(span=20).mean()

    # Bollinger Bands
    std = df["Close"].rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std
    df["BB_Lower"] = df["SMA20"] - 2 * std

    # RSI
    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    rs = gain.rolling(14).mean() / loss.rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9).mean()

    # ATR
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    df["ATR"] = tr.rolling(14).mean()

    # VWAP
    df["VWAP"] = (
        (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3)
        .cumsum() / df["Volume"].cumsum()
    )

    # Volume MA
    df["VOL_MA20"] = df["Volume"].rolling(20).mean()

    return df

def detect_anomalies(df):
    returns = df["Close"].pct_change().dropna()
    model = IsolationForest(contamination=0.03, random_state=42)
    preds = model.fit_predict(returns.values.reshape(-1, 1))
    df["Anomaly"] = 1
    df.loc[returns.index, "Anomaly"] = preds
    return df

# --------------------------------------------------
# Prophet Forecast
# --------------------------------------------------
def prepare_prophet(df):
    p = df["Close"].reset_index()
    p.columns = ["ds", "y"]
    return p

@st.cache_resource
def train_prophet(p):
    model = Prophet(
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05
    )
    model.fit(p)
    return model

def forecast_price(df, days):
    p = prepare_prophet(df)
    model = train_prophet(p)
    future = model.make_future_dataframe(periods=days)
    return model.predict(future)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------
st.sidebar.title("⚙️ Controls")

tickers = st.sidebar.text_input(
    "Tickers (comma separated)", "AAPL,MSFT"
).upper().split(",")

period = st.sidebar.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=1)
forecast_days = st.sidebar.slider("Forecast Days", 7, 180, 60)

price_indicators = st.sidebar.multiselect(
    "Price Indicators",
    ["SMA20", "SMA50", "EMA20", "BB_Upper", "BB_Lower", "VWAP"],
    default=["SMA20", "SMA50"]
)

# --------------------------------------------------
# Tabs
# --------------------------------------------------
tabs = st.tabs([
    "📈 Price + Indicators",
    "📊 RSI & MACD",
    "🔮 Forecast",
    "🏦 Fundamentals",
    "📥 Export"
])

# --------------------------------------------------
# Price + Indicators
# --------------------------------------------------
with tabs[0]:
    fig = go.Figure()

    for t in tickers:
        df = load_stock_data(t.strip(), period)
        if df is None:
            continue

        df = indicators(df)
        df = detect_anomalies(df)

        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            name=f"{t.strip()} Close",
            line=dict(width=2)
        ))

        for ind in price_indicators:
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[ind],
                name=f"{t.strip()} {ind}",
                line=dict(dash="dot")
            ))

        anomalies = df[df["Anomaly"] == -1]
        fig.add_trace(go.Scatter(
            x=anomalies.index,
            y=anomalies["Close"],
            mode="markers",
            marker=dict(color="red", size=6),
            name=f"{t.strip()} Anomaly"
        ))

    fig.update_layout(
        height=520,
        title="Price with Advanced Technical Indicators",
        xaxis_rangeslider_visible=False
    )
    st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# RSI & MACD
# --------------------------------------------------
with tabs[1]:
    for t in tickers:
        df = load_stock_data(t.strip(), period)
        if df is None:
            continue
        df = indicators(df)

        st.subheader(f"{t.strip()} RSI")
        rsi_fig = go.Figure()
        rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
        rsi_fig.add_hline(y=70, line_dash="dash")
        rsi_fig.add_hline(y=30, line_dash="dash")
        rsi_fig.update_layout(height=260)
        st.plotly_chart(rsi_fig, width="stretch")

        st.subheader(f"{t.strip()} MACD")
        macd_fig = go.Figure()
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
        macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal"))
        macd_fig.update_layout(height=260)
        st.plotly_chart(macd_fig, width="stretch")

# --------------------------------------------------
# Forecast
# --------------------------------------------------
with tabs[2]:
    for t in tickers:
        df = load_stock_data(t.strip(), period)
        if df is None:
            continue
        fc = forecast_price(df, forecast_days)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"], name="History"))
        fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast"))
        fig.update_layout(title=f"{t.strip()} Price Forecast", height=420)
        st.plotly_chart(fig, width="stretch")

# --------------------------------------------------
# Fundamentals (Arrow-safe)
# --------------------------------------------------
with tabs[3]:
    for t in tickers:
        fundamentals = load_fundamentals(t.strip())
        fdf = (
            pd.DataFrame(fundamentals, index=["Value"])
            .T.reset_index()
            .rename(columns={"index": "Metric"})
        )
        fdf["Value"] = fdf["Value"].astype(str)
        st.subheader(f"{t.strip()} Fundamentals")
        st.dataframe(fdf, width="stretch")

# --------------------------------------------------
# Export
# --------------------------------------------------
with tabs[4]:
    export_frames = []
    for t in tickers:
        df = load_stock_data(t.strip(), period)
        if df is not None:
            df["Ticker"] = t.strip()
            export_frames.append(df)

    if export_frames:
        export_df = pd.concat(export_frames)
        st.download_button(
            "⬇️ Download All Data (CSV)",
            export_df.to_csv().encode("utf-8"),
            file_name=f"market_data_{datetime.now().date()}.csv",
            mime="text/csv"
        )


