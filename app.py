import streamlit as st
import os
import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
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
    
@st.cache_data(ttl=3600)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    # Fake a real Google Chrome web browser request
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        # Use requests to bypass the Wikipedia block
        response = requests.get(url, headers=headers, timeout=10)
        
        # Read the HTML text directly from our successful request
        tables = pd.read_html(response.text)
        df = tables[0]
        
        tickers = df['Symbol'].str.replace('.', '-', regex=False).tolist()
        return tickers
        
    except Exception as e:
        st.error(f"Failed to scrape S&P 500 tickers: {e}")
        # Fallback to a tiny static list just in case Wikipedia is entirely down
        return ["SPY", "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK-B", "JNJ"]
    
def calculate_swing_metrics(df_close, df_high, df_low, df_volume, tickers_info):
    results = []
    
    for ticker in df_close.columns:
        close = df_close[ticker].dropna()
        volume = df_volume[ticker].dropna()
        
        if len(close) < 60 or len(volume) < 20: 
            continue
            
        current_price = float(close.iloc[-1])
        current_volume = float(volume.iloc[-1])
        
        # 1. RSI (14-day)
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        current_rsi = float(rsi.iloc[-1])
        
        # 2. Distance from 50 SMA (%)
        sma50 = close.rolling(window=50).mean()
        dist_sma50 = float(((current_price - sma50.iloc[-1]) / sma50.iloc[-1]) * 100)
        
        # 3. MACD Histogram (12, 26, 9)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = float((macd - signal).iloc[-1])
        
        # 4. Volatility Proxy (Standard Deviation annualized)
        volatility_30d = float((close.pct_change().rolling(20).std() * np.sqrt(252)).iloc[-1] * 100)
        
        # 5. Momentum (Rate of Change over 10 days)
        roc_10d = float(((current_price - close.iloc[-10]) / close.iloc[-10]) * 100)
        
        # NEW CRITERIA 6: Volume Surge (Current Volume / 20-Day Average Volume)
        avg_volume_20d = volume.rolling(window=20).mean().iloc[-1]
        volume_ratio = current_volume / (avg_volume_20d + 1e-9)
        
        # NEW CRITERIA 7: Float Turnover % (Current Volume / Share Float)
        # Safely fetch the cached float from the tickers_info dictionary
        share_float = tickers_info.get(ticker, {}).get('float', None)
        if share_float and share_float > 0:
            float_turnover = (current_volume / share_float) * 100
        else:
            float_turnover = 0.0

        # --- UPGRADED SWING SCORING ALGORITHM (MAX SCORE = 15) ---
        score = 0
        if 45 <= current_rsi <= 65: score += 3     # Healthy momentum zone
        if -3 <= dist_sma50 <= 5: score += 3       # Pulling back near support
        if macd_hist > 0: score += 2              # Bullish MACD crossover
        if volatility_30d > 25: score += 2        # High volatility setup
        if roc_10d > 0: score += 1                # Positive short-term velocity
        
        # Scoring for Volume Surge
        if volume_ratio >= 2.0: score += 2        # Volume is double the 20-day average
        elif volume_ratio >= 1.2: score += 1      # Volume is 20% above average
        
        # Scoring for Float Turnover
        if float_turnover >= 5.0: score += 2      # Heavy hand-over-hand stock turnover (>5% of float)
        elif float_turnover >= 1.5: score += 1    # Healthy active trading turnover (>1.5% of float)
        
        results.append({
            "Ticker": ticker, 
            "Price": round(current_price, 2), 
            "RSI (14d)": round(current_rsi, 1),
            "Dist from 50SMA (%)": round(dist_sma50, 2), 
            "MACD Hist": round(macd_hist, 3),
            "Volatility %": round(volatility_30d, 1), 
            "Vol Surge Ratio": round(volume_ratio, 2),
            "Float Turnover %": round(float_turnover, 2),
            "Swing Score": score
        })
    return pd.DataFrame(results)

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

        tech_sub_sectors = {
            "SOXX": "Semiconductors",
            "IGV": "Software",
            "CIBR": "Cybersecurity",
            "SKYY": "Cloud Computing",
            "IYW": "Tech Hardware"
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
        st.dataframe(daily_data, width='stretch', height=250)

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

        # ==============================
        # 5. TECH SUB-SECTOR ROTATION
        # ==============================
        st.header("🔬 Technology Sub-Sector Deep Dive")
        st.write("Tracking the inner momentum of the Tech sector to see if chips, software, or security are leading.")

        tech_trends = {}

        with st.spinner("Fetching sub-sector data..."):
            for ticker, name in tech_sub_sectors.items():
                df_sub = yf.download(ticker, period="30d", auto_adjust=True, progress=False)
                
                # 1. Safety check: Skip this ticker if yfinance returned an empty DataFrame
                if df_sub.empty:
                    st.warning(f"⚠️ Could not fetch data for {ticker} ({name}). Skipping...")
                    continue
                    
                close_series = df_sub["Close"]
                
                if isinstance(close_series, pd.DataFrame):
                    close_series = close_series.iloc[:, 0]
                    
                # 2. Double-check that the series actually has data points left
                if len(close_series.dropna()) > 0:
                    # Safe to use .iloc[0] now because we verified data exists!
                    tech_trends[name] = close_series / close_series.dropna().iloc[0] * 100
                else:
                    continue

            # 3. Only attempt to build the chart if at least one sub-sector succeeded
            if tech_trends:
                df_tech_trends = pd.DataFrame(tech_trends)

                fig5 = plt.figure(figsize=(14, 6))
                for column in df_tech_trends.columns:
                    plt.plot(df_tech_trends.index, df_tech_trends[column], label=column, linewidth=2)

                plt.title("Technology Sub-Sector Performance (Base=100)", fontsize=14)
                plt.ylabel("Normalized Return")
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.tight_layout()
                
                st.pyplot(fig5)
            else:
                st.error("❌ Failed to fetch data for all sub-sectors. Check your internet connection or tickers.")

        # ------------------------------------------
        # SECTION 4: HIGH-SPEED SWING PLAN SCANNER
        # ------------------------------------------
        st.header("🎯 S&P 500 Alpha Swing Trading Signals")
        st.write("Scrapes and filters alpha setups instantly using custom momentum-reversion weights.")
        
        tickers_sp500 = get_sp500_tickers()
        
        # Single vectorized web call to get OHLCV matrix data for all stocks
        sp500_raw = yf.download(tickers_sp500, period="90d", interval="1d", group_by='column', auto_adjust=True, progress=False)
        
        # Step to compile a fast data dictionary of stock float measurements
        # We leverage yf.Tickers for multi-threaded fast properties download
        with st.spinner("Analyzing share structures and float parameters..."):
            tickers_object = yf.Tickers(" ".join(tickers_sp500))
            tickers_info = {}
            for t in tickers_sp500:
                try:
                    # Capture broad float or default shares outstanding seamlessly
                    info = tickers_object.tickers[t].info
                    tickers_info[t] = {
                        'float': info.get('floatShares', info.get('sharesOutstanding', None))
                    }
                except:
                    tickers_info[t] = {'float': None}
        
        # Run calculations using our brand new multi-criteria framework
        scanner_results = calculate_swing_metrics(
            sp500_raw['Close'], 
            sp500_raw['High'], 
            sp500_raw['Low'], 
            sp500_raw['Volume'],
            tickers_info
        )
        
        # Increase visibility scope to capture more than 10 high scorers
        top_setups = scanner_results.sort_values(by="Swing Score", ascending=False).head(25).reset_index(drop=True)
        
        st.subheader("🔥 Top Structural Technical Swing Profiles")
        st.dataframe(
            top_setups.style.background_gradient(subset=["Swing Score"], cmap="YlGn")
                            .background_gradient(subset=["RSI (14d)"], cmap="bwr", vmin=30, vmax=70)
                            .background_gradient(subset=["Vol Surge Ratio"], cmap="Purples")
                            .background_gradient(subset=["Float Turnover %"], cmap="Oranges"),
            width='stretch'
        )
        st.success("Analysis and Advanced Scans Finished Successfully!")

    st.success("Analysis Complete!")