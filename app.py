import streamlit as st
import os
import datetime
import requests
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from io import StringIO

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
def get_sp500_sectors_map():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        html_data = StringIO(response.text)
        tables = pd.read_html(html_data)
        df = tables[0]
        
        # Clean up tickers for yfinance compatibility
        df['Clean_Symbol'] = df['Symbol'].str.replace('.', '-', regex=False)
        
        # Create a dictionary mapping: {'AAPL': 'Information Technology', 'JPM': 'Financials', ...}
        sector_map = dict(zip(df['Clean_Symbol'], df['GICS Sector']))
        return sector_map
        
    except Exception as e:
        st.error(f"Failed to scrape S&P 500 sectors: {e}")
        # Default fallback map for mega-caps
        return {
            "SPY": "Index", "AAPL": "Information Technology", "MSFT": "Information Technology", 
            "AMZN": "Consumer Discretionary", "NVDA": "Information Technology", "GOOGL": "Communication Services", 
            "META": "Communication Services", "TSLA": "Consumer Discretionary", "BRK-B": "Financials", "JNJ": "Health Care"
        }
    
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

def calculate_momentum_leaders(df_close, sectors_map):
    leaders_data = []
    
    # We need to make sure we look back far enough for YTD calculation
    # In mid-2026, the first trading day of the year was Jan 2, 2026
    for ticker in df_close.columns:
        close = df_close[ticker].dropna()
        if len(close) < 5:  # Skip if not enough history even for a week
            continue
            
        current_price = float(close.iloc[-1])
        
        # 1-Week Return (approx. 5 trading days)
        w1_pct = float(((current_price - close.iloc[-5]) / close.iloc[-5]) * 100) if len(close) >= 5 else 0.0
        
        # 1-Month Return (approx. 21 trading days)
        m1_pct = float(((current_price - close.iloc[-21]) / close.iloc[-21]) * 100) if len(close) >= 21 else 0.0
        
        # Year-to-Date Return 
        # Dynamically find the earliest date belonging to the current year (2026)
        ytd_dates = close.index[close.index.year == 2026]
        if len(ytd_dates) > 0:
            ytd_start_price = float(close.loc[ytd_dates[0]])
            ytd_pct = float(((current_price - ytd_start_price) / ytd_start_price) * 100)
        else:
            ytd_pct = 0.0
            
        leaders_data.append({
            "Ticker": ticker,
            "Sector": sectors_map.get(ticker, "Unknown"),
            "Price": round(current_price, 2),
            "1-Week %": round(w1_pct, 2),
            "1-Month %": round(m1_pct, 2),
            "YTD %": round(ytd_pct, 2)
        })
        
    df_leaders = pd.DataFrame(leaders_data)
    
    # Extract top 10 for each category
    top_week = df_leaders.sort_values(by="1-Week %", ascending=False).head(10)[["Ticker", "Sector", "Price", "1-Week %"]].reset_index(drop=True)
    top_month = df_leaders.sort_values(by="1-Month %", ascending=False).head(10)[["Ticker", "Sector", "Price", "1-Month %"]].reset_index(drop=True)
    top_ytd = df_leaders.sort_values(by="1-Based %" if "1-Based %" in df_leaders.columns else "YTD %", ascending=False).head(10)[["Ticker", "Sector", "Price", "YTD %"]].reset_index(drop=True)
    
    return top_week, top_month, top_ytd

def calculate_rsi_reversals(df_close, df_volume, sectors_map):
    reversal_data = []
    
    for ticker in df_close.columns:
        close = df_close[ticker].dropna()
        volume = df_volume[ticker].dropna()
        
        # We need at least 16 days of data to compute rolling 14-day RSI up to today
        if len(close) < 16 or len(volume) < 2:
            continue
            
        # --- CALCULATE CONTINUOUS RSI SERIES ---
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-9)
        rsi_series = 100 - (100 / (1 + rs))
        
        if len(rsi_series.dropna()) < 2:
            continue
            
        # Isolate yesterday and today parameters
        rsi_yesterday = float(rsi_series.iloc[-2])
        rsi_today = float(rsi_series.iloc[-1])
        
        # --- CONDITIONAL TRIGGER CONDITION ---
        # Yesterday was oversold (< 30), Today crossed back up (> 30)
        if rsi_yesterday < 30 and rsi_today > 30:
            vol_yesterday = float(volume.iloc[-2])
            vol_today = float(volume.iloc[-1])
            
            # Volume surge ratio comparison
            vol_surge_ratio = vol_today / (vol_yesterday + 1e-9)
            
            reversal_data.append({
                "Ticker": ticker,
                "Sector": sectors_map.get(ticker, "Unknown"),
                "Current Price": round(float(close.iloc[-1]), 2),
                "Yesterday RSI": round(rsi_yesterday, 1),
                "Today RSI": round(rsi_today, 1),
                "Volume Surge Ratio": round(vol_surge_ratio, 2)
            })
            
    if not reversal_data:
        return pd.DataFrame(columns=["Ticker", "Sector", "Current Price", "Yesterday RSI", "Today RSI", "Volume Surge Ratio"])
        
    df_reversals = pd.DataFrame(reversal_data)
    
    # Sort by the highest volume breakout ratio and select the top 10
    top_10_reversals = df_reversals.sort_values(by="Volume Surge Ratio", ascending=False).head(10).reset_index(drop=True)
    return top_10_reversals


# ==============================
# TRIGGER BUTTON
# ==============================
if st.button("Hello", type="primary"):
    
    with st.spinner("Fetching market data and building charts..."):
        
        tickers = {
        "SPY": "SPY", "RSP": "RSP", "IWM": "IWM", "VIX": "^VIX",
        "Gold": "GC=F", "Oil": "CL=F", "TLT": "TLT", "HYG": "HYG",
        "KRE": "KRE", "DXY": "DX-Y.NYB",
        "US10Y": "^TNX"
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

       # 🏦 Regional Banking Health Ratio (KRE divided by Broad Financials XLF)
        # Note: We can download XLF directly here or use SPY as the denominator base
        daily_data["KRE/SPY"] = daily_data["KRE"] / daily_data["SPY"]

        # 🛡️ Synthetic Credit Default Swap (CDS) / Credit Stress Proxy 
        # As credit risk rises, Stocks fall and High-Yield Bonds get crushed relative to safe Treasuries
        daily_data["Credit_Stress_Proxy"] = daily_data["TLT"] / daily_data["HYG"] 
        
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

        # Plot 3: 10-Year Treasury Yield Trend Line
        st.subheader("🏛️ US 10-Year Treasury Yield Trend")
        st.write("Tracks the baseline cost of capital. Rising yields put pressure on high-multiple growth stocks.")
        
        if "US10Y" in daily_data.columns:
            # Create actual yield percentages (yfinance stores 4.25% as 42.5)
            yield_percentage = daily_data["US10Y"]
            
            fig4, ax4 = plt.subplots(figsize=(14, 4))
            ax4.plot(daily_data["Date"], yield_percentage, color="crimson", linewidth=2.5, label="US 10Y Yield (%)")
            
            ax4.set_title("US 10-Year Treasury Yield Momentum (30-Day Window)", fontsize=13, fontweight='bold')
            ax4.set_ylabel("Yield (%)", fontsize=11)
            ax4.set_xlabel("Date", fontsize=11)
            ax4.grid(True, linestyle="--", alpha=0.6)
            
            # Format the latest metric reading cleanly on top of the axis framework
            latest_yield = yield_percentage.dropna().iloc[-1]
            ax4.axhline(latest_yield, color="gray", linestyle=":", alpha=0.7)
            ax4.text(daily_data["Date"].iloc[-1], latest_yield, f"  {latest_yield:.2f}%", 
                     va='center', ha='left', color='crimson', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig4)
        else:
            st.error("Treasury Yield data unavailable for charting.")

        # 1. Macro Dashboard
        fig, ax = plt.subplots(2, 2, figsize=(14, 8))
        ax[0,0].plot(daily_data["Date"], daily_data["RSP/SPY"], label="RSP/SPY")
        ax[0,0].plot(daily_data["Date"], daily_data["IWM/SPY"], label="IWM/SPY")
        ax[0,0].set_title("Breadth")
        ax[0,0].legend(); ax[0,0].grid()

        # Quadrant 2: Regional Banking System Health (KRE)
        ax[0,1].plot(daily_data["Date"], daily_data["KRE/SPY"], color="teal")
        ax[0,1].set_title("Banking System Liquidity (KRE/SPY)"); ax[0,1].grid()

        ax[1,0].plot(daily_data["Date"], daily_data["VIX"])
        ax[1,0].set_title("VIX"); ax[1,0].grid()

        # Quadrant 4: Credit Default / Systemic Risk Proxy (Synthetic CDS)
        ax[1,1].plot(daily_data["Date"], daily_data["Credit_Stress_Proxy"], color="purple")
        ax[1,1].set_title("Synthetic CDS / Credit Stress (TLT/HYG)"); ax[1,1].grid()

        plt.tight_layout()
        st.pyplot(fig)  # <-- Instead of plt.savefig

        # 2. Cross Asset Performance
        # Updated asset tracking list
        assets = ["SPY", "IWM", "TLT", "Gold", "Oil", "DXY", "HYG", "KRE"]
        color_map = {
            "SPY": "blue", "IWM": "orange", "TLT": "green", "Gold": "gold", 
            "Oil": "black", "DXY": "purple", "HYG": "red", "KRE": "teal"
        }
        
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

        # FIX: Added .dropna() validation to prevent vanishing curves
        US10Y_norm = df_risk["US10Y"] / df_risk["US10Y"].dropna().iloc[0] * 100 if not df_risk["US10Y"].dropna().empty else np.nan

        fig4, ax4 = plt.subplots(2, 2, figsize=(14, 8))
        ax4[0, 0].plot(df_risk.index, spy_norm, label="SPY", color="#1f77b4")
        ax4[0, 0].plot(df_risk.index, tlt_norm, label="TLT", color="#2ca02c")
        ax4[0, 0].set_title("SPY vs TLT (Base=100)"); ax4[0, 0].legend(); ax4[0, 0].grid(True)

        ax4[0, 1].plot(df_risk.index, df_risk["SPY"], color="#1f77b4")
        ax4[0, 1].plot(df_risk.index, df_risk["IWM"], color="#ff7f0e")
        ax4[0, 1].set_title("SPY vs IWM"); ax4[0, 1].legend(); ax4[0, 1].grid(True)

        ax4[1, 0].plot(df_risk.index, df_risk["IWM_SPY"], color="#9467bd")
        ax4[1, 0].set_title("IWM / SPY Ratio"); ax4[1, 0].grid(True)

        ax4[1, 1].plot(df_risk.index, tlt_norm, label="TLT Price", color="#d62728")
        if not isinstance(US10Y_norm, float):
            ax4[1, 1].plot(df_risk.index, US10Y_norm, label="US10Y Yield", color="#2c36a0")
        ax4[1, 1].set_title("US10Y Yield vs TLT Valuation Velocity (Base=100)"); ax4[1, 1].legend(); ax4[1, 1].grid(True)
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
        
        # 1. Fetch the mapping dictionary
        sectors_map = get_sp500_sectors_map()
        tickers_sp500 = list(sectors_map.keys())
        
        # Vectorized download for market metrics
        sp500_raw = yf.download(tickers_sp500, period="1y", interval="1d", group_by='column', auto_adjust=True, progress=False)
        
        # Analyze share structures for float details
        with st.spinner("Analyzing share structures and float parameters..."):
            tickers_object = yf.Tickers(" ".join(tickers_sp500))
            tickers_info = {}
            for t in tickers_sp500:
                try:
                    info = tickers_object.tickers[t].info
                    tickers_info[t] = {
                        'float': info.get('floatShares', info.get('sharesOutstanding', None))
                    }
                except:
                    tickers_info[t] = {'float': None}
        
        # Run calculations
        scanner_results = calculate_swing_metrics(
            sp500_raw['Close'], 
            sp500_raw['High'], 
            sp500_raw['Low'], 
            sp500_raw['Volume'],
            tickers_info
        )
        
        # 2. INJECT SECTOR INFORMATION NATIVELY
        scanner_results['Sector'] = scanner_results['Ticker'].map(sectors_map)
        
        # Re-order columns nicely so Sector sits right next to the Ticker name
        column_order = ["Ticker", "Sector", "Price", "RSI (14d)", "Dist from 50SMA (%)", "MACD Hist", "Volatility %", "Vol Surge Ratio", "Float Turnover %", "Swing Score"]
        scanner_results = scanner_results[column_order]
        
        # Sort and take top 25 setups
        top_setups = scanner_results.sort_values(by="Swing Score", ascending=False).head(25).reset_index(drop=True)
        
        st.subheader("🔥 Top Structural Technical Swing Profiles")
        st.dataframe(
            top_setups.style.background_gradient(subset=["Swing Score"], cmap="YlGn")
                            .background_gradient(subset=["RSI (14d)"], cmap="bwr", vmin=30, vmax=70)
                            .background_gradient(subset=["Vol Surge Ratio"], cmap="Purples")
                            .background_gradient(subset=["Float Turnover %"], cmap="Oranges"),
            width='stretch'
        )

        # ------------------------------------------
        # SECTION 5: MULTI-TIMEFRAME MOMENTUM LEADERS
        # ------------------------------------------
        st.write("---")
        st.header("⚡ S&P 500 Absolute Momentum Leaders")
        st.write("Cross-referencing short, medium, and long-term velocity vectors to target leading industry flows.")
        
        # Calculate the vectors
        top_week, top_month, top_ytd = calculate_momentum_leaders(sp500_raw['Close'], sectors_map)
        
        # Render the UI Columns
        col_w, col_m, col_y = st.columns(3)
        
        with col_w:
            st.subheader("🏃‍♂️ Short Term (1-Week)")
            st.dataframe(
                top_week.style.background_gradient(subset=["1-Week %"], cmap="Greens"),
                width='stretch'
            )
            
        with col_m:
            st.subheader("📈 Intermediate (1-Month)")
            st.dataframe(
                top_month.style.background_gradient(subset=["1-Month %"], cmap="Blues"),
                width='stretch'
            )
            
        with col_y:
            st.subheader("🏆 Structural (Year-to-Date)")
            st.dataframe(
                top_ytd.style.background_gradient(subset=["YTD %"], cmap="Purples"),
                width='stretch'
            )

        
        # ==========================================
        # SECTION 6: RSI OVERSOLD REVERSAL BREAKOUTS
        # ==========================================
        st.write("---")
        st.header("🔄 S&P 500 Oversold RSI Reversals")
        st.write("Identifies stocks that were oversold yesterday (RSI < 30) but broke back up today (RSI > 30), ranked by the highest relative volume surge.")
        
        # Run calculation engine
        top_10_reversals = calculate_rsi_reversals(sp500_raw['Close'], sp500_raw['Volume'], sectors_map)
        
        if not top_10_reversals.empty:
            st.dataframe(
                top_10_reversals.style.background_gradient(subset=["Volume Surge Ratio"], cmap="YlOrRd")
                                      .background_gradient(subset=["Today RSI"], cmap="Greens", vmin=30, vmax=40),
                width='stretch'
            )
        else:
            st.info("No stocks currently match the RSI reversal parameters today. (No assets crossed from under 30 to above 30 on this session).")


        st.success("Analysis and Advanced Scans Finished Successfully!")

    st.success("Analysis Complete!")