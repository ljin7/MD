import streamlit as st
import os
import datetime
import requests
import yfinance as yf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# Optional: Keep fear_greed if you have the local file/package, 
# otherwise we rely on the API call you wrote.
try:
    import fear_greed
except ImportError:
    fear_greed = None

# Streamlit page configuration
st.set_page_config(page_title="MD", layout="wide")
st.title("MD")
st.write("Click")

# ==============================
# HELPERS (Kept exactly from your script)
# ==============================
def get_close_series(ticker, name):
    df = yf.download(ticker, period="30d", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    close.name = name
    return close

def get_latest_price(ticker):
    df = yf.download(ticker, period="10d", interval="1d", auto_adjust=True, progress=False)
    if df.empty: return None
    close = df["Close"]
    if isinstance(close, pd.DataFrame): return float(close.iloc[-1, 0])
    return float(close.iloc[-1])

def get_fear_greed():
    try:
        url = "https://api.alternative.me/fng/?limit=1"
        r = requests.get(url, timeout=10)
        data = r.json()
        return float(data["data"][0]["value"])
    except Exception as e:
        st.error(f"FearGreed API error: {e}")
        return None

def safe_ratio(a, b):
    if a is None or b is None or b == 0: return None
    return a / b

def get_hy_spread():
    try:
        url = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAMLH0A0HYM2"
        df = pd.read_csv(url)
        last_valid = df.iloc[:, 1].dropna()
        if len(last_valid) > 0: return float(last_valid.iloc[-1])
        return None
    except Exception as e:
        st.error(f"HY Spread error: {e}")
        return None

# ==============================
# TRIGGER BUTTON
# ==============================
if st.button("Hello", type="primary"):
    
    with st.spinner("Fetching market data and building charts..."):
        
        tickers = {
            "SPY": "SPY", "RSP": "RSP", "IWM": "IWM", "VIX": "^VIX",
            "Gold": "GC=F", "Oil": "CL=F", "TLT": "TLT", "HYG": "HYG",
            "JNK": "JNK", "DXY": "DX-Y.NYB"
        }
        
        sector_etfs = {
            "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
            "XLV": "Healthcare", "XLY": "Consumer Discretionary",
            "XLP": "Consumer Staples", "XLU": "Utilities", "XLI": "Industrials",
            "XLB": "Materials", "XLRE": "Real Estate", "XLC": "Communication Services"
        }

        # --- BUILD DAILY DATA ---
        spy = get_close_series("SPY", "SPY")
        daily_data = spy.to_frame()

        for k, v in tickers.items():
            if k == "SPY": continue
            s = get_close_series(v, k)
            if s is not None:
                daily_data = daily_data.join(s, how="left")

        daily_data = daily_data.ffill().reset_index()
        if "index" in daily_data.columns:
            daily_data.rename(columns={"index": "Date"}, inplace=True)
        daily_data["Date"] = pd.to_datetime(daily_data["Date"])

        # --- RATIOS & SNAPSHOTS ---
        daily_data["RSP/SPY"] = daily_data["RSP"] / daily_data["SPY"]
        daily_data["IWM/SPY"] = daily_data["IWM"] / daily_data["SPY"]
        daily_data["HYG/JNK"] = daily_data["HYG"] / daily_data["JNK"]
        daily_data["Gold/Oil"] = daily_data["Gold"] / daily_data["Oil"]

        latest = {k: get_latest_price(v) for k, v in tickers.items()}

        data_feargreed = fear_greed.get()

        fg_val = data_feargreed['score']
        ra_val = data_feargreed['rating']
        hy_val = get_hy_spread()

        # ==============================
        # DISPLAY TABLES & METRICS
        # ==============================
        st.header("📋 Latest Market Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric(label="Fear & Greed Index Score", value=fg_val if fg_val else "N/A")
        col2.metric(label="High Yield Spread", value=f"{hy_val}%" if hy_val else "N/A")
        col3.metric(label="rating", value=ra_val if ra_val else "N/A")

        st.subheader("Recent Daily Data Table")
        # Displaying the interactive dataframe directly on the webpage
        st.dataframe(daily_data, use_container_width=True)

        # ==============================
        # DISPLAY RENDERED PLOTS
        # ==============================
        st.header("📉 Dashboard Visualizations")

        # 1. Macro Dashboard
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        ax[0,0].plot(daily_data["Date"], daily_data["RSP/SPY"], label="RSP/SPY")
        ax[0,0].plot(daily_data["Date"], daily_data["IWM/SPY"], label="IWM/SPY")
        ax[0,0].set_title("Breadth")
        ax[0,0].legend(); ax[0,0].grid()

        ax[0,1].plot(daily_data["Date"], daily_data["HYG/JNK"])
        ax[0,1].set_title("HYG/JNK"); ax[0,1].grid()

        ax[1,0].plot(daily_data["Date"], daily_data["VIX"])
        ax[1,0].set_title("VIX"); ax[1,0].grid()

        ax[1,1].plot(daily_data["Date"], daily_data["Gold/Oil"])
        ax[1,1].set_title("Gold/Oil"); ax[1,1].grid()
        plt.tight_layout()
        st.pyplot(fig)  # <-- Instead of plt.savefig

        # 2. Cross Asset Performance
        assets = ["SPY","IWM","TLT","Gold","Oil","DXY","HYG"]
        color_map = {"SPY": "blue", "IWM": "orange", "TLT": "green", "Gold": "gold", "Oil": "black", "DXY": "purple", "HYG": "red"}
        
        df_perf = daily_data.set_index("Date")
        perf = pd.DataFrame(index=df_perf.index)
        for a in assets:
            if a in df_perf.columns:
                series = df_perf[a].ffill()
                if not series.dropna().empty:
                    perf[a] = series / series.dropna().iloc[0] * 100

        fig2 = plt.figure(figsize=(14, 6))
        for a in perf.columns:
            plt.plot(perf.index, perf[a], label=a, color=color_map.get(a, None), linewidth=2)
        plt.title("Cross Asset (Base=100)")
        plt.legend(); plt.grid(True); plt.tight_layout()
        st.pyplot(fig2)

        # 3. Sector Rotation
        sector = {}
        for k, v in sector_etfs.items():
            df_sec = yf.download(k, period="30d", auto_adjust=True, progress=False)
            c = df_sec["Close"]
            if isinstance(c, pd.DataFrame): c = c.iloc[:,0]
            sector[v] = c / c.iloc[0] * 100
        sd = pd.DataFrame(sector)

        fig3 = plt.figure(figsize=(14, 6))
        for c in sd.columns:
            plt.plot(sd.index, sd[c], label=c)
        plt.title("Sector Rotation")
        plt.legend(bbox_to_anchor=(1.02,1), loc='upper left')
        plt.grid(); plt.tight_layout()
        st.pyplot(fig3)

        # 4. Risk Dashboard
        df_risk = daily_data.set_index("Date").sort_index().ffill()
        df_risk["IWM_SPY"] = df_risk["IWM"] / df_risk["SPY"]
        spy_norm = df_risk["SPY"] / df_risk["SPY"].iloc[0] * 100
        tlt_norm = df_risk["TLT"] / df_risk["TLT"].iloc[0] * 100

        fig4, ax4 = plt.subplots(2, 2, figsize=(14, 8))
        ax4[0, 0].plot(df_risk.index, spy_norm, label="SPY", color="#1f77b4")
        ax4[0, 0].plot(df_risk.index, tlt_norm, label="TLT", color="#2ca02c")
        ax4[0, 0].set_title("SPY vs TLT (Base=100)"); ax4[0, 0].legend(); ax4[0, 0].grid(True)

        ax4[0, 1].plot(df_risk.index, df_risk["SPY"], color="#1f77b4")
        ax4[0, 1].plot(df_risk.index, df_risk["IWM"], color="#ff7f0e")
        ax4[0, 1].set_title("SPY vs IWM"); ax4[0, 1].grid(True)

        ax4[1, 0].plot(df_risk.index, df_risk["IWM_SPY"], color="#9467bd")
        ax4[1, 0].set_title("IWM / SPY Ratio"); ax4[1, 0].grid(True)

        ax4[1, 1].plot(df_risk.index, df_risk["VIX"], color="#d62728")
        ax4[1, 1].set_title("VIX"); ax4[1, 1].grid(True)
        plt.tight_layout()
        st.pyplot(fig4)
        
        st.success("Analysis Complete!")