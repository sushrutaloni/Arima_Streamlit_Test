import streamlit as st
import yfinance as yf
import datetime as d
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# APP TITLE
# ------------------------------
st.set_page_config(page_title="Stock Forecast Dashboard", layout="wide")
st.title("📈 IT Stock Forecast Dashboard (India)")
st.markdown("Analyze and forecast Indian IT giants using ARIMA model.")

# ------------------------------
# DATA DOWNLOAD
# ------------------------------
@st.cache_data
def load_data():
    s = d.datetime(2024, 1, 1)
    e = d.datetime(2025, 11, 12)

    tickers = {
        "TCS": "TCS.NS",
        "INFOSYS": "INFY.NS",
        "WIPRO": "WIPRO.NS",
        "HCLTECH": "HCLTECH.NS",
        "LTIM": "LTIM.NS",
        "PERSISTENT": "PERSISTENT.NS"
    }

    data_frames = []
    for stock, ticker in tickers.items():
        df = yf.download(ticker, start=s, end=e)
        df.columns = df.columns.get_level_values(0)
        df = df.reset_index()
        df["Stock"] = stock
        data_frames.append(df)

    final_df = pd.concat(data_frames, axis=0)
    final_df = final_df.set_index("Date")
    return final_df

df = load_data()

# ------------------------------
# STOCK SELECTION
# ------------------------------
st.sidebar.header("⚙️ Controls")
stock_list = df["Stock"].unique().tolist()
selected_stock = st.sidebar.selectbox("Select Stock", stock_list)

st.subheader(f"📊 Historical Prices — {selected_stock}")
st.write("Showing data from Jan 2024 to Nov 2025")

st_data = df[df["Stock"] == selected_stock][["Close"]]
st_data["Returns"] = st_data["Close"].pct_change()
st_data.dropna(inplace=True)

# ------------------------------
# ADF TEST FUNCTION
# ------------------------------
def adf_test(series):
    result = adfuller(series.dropna())
    return {
        "ADF Statistic": result[0],
        "P-value": result[1],
        "Stationary": "Yes ✅" if result[1] <= 0.05 else "No ❌"
    }

# ------------------------------
# CHECK STATIONARITY
# ------------------------------
with st.expander("📈 Stationarity Check (ADF Test)"):
    col1, col2 = st.columns(2)

    adf_close = adf_test(st_data["Close"])
    adf_diff = adf_test(st_data["Close"].diff().dropna())

    with col1:
        st.write("**Original Series (Close)**")
        st.table(pd.DataFrame([adf_close]))

    with col2:
        st.write("**Differenced Series (Close_Diff)**")
        st.table(pd.DataFrame([adf_diff]))

# ------------------------------
# ARIMA FORECASTING
# ------------------------------
steps = st.sidebar.slider("Forecast Steps (Days)", 5, 30, 10)

model = ARIMA(st_data["Close"], order=(5, 1, 0))
model_fit = model.fit()
forecast = model_fit.forecast(steps=steps)

future_dates = pd.date_range(start=st_data.index[-1], periods=steps + 1, freq="B")[1:]

# ------------------------------
# PLOTLY VISUALIZATION
# ------------------------------
fig = go.Figure()

# Actual Prices
fig.add_trace(go.Scatter(
    x=st_data.index,
    y=st_data["Close"],
    mode="lines",
    name="Actual Price",
    line=dict(color="royalblue")
))

# Forecasted Prices
fig.add_trace(go.Scatter(
    x=future_dates,
    y=forecast,
    mode="lines+markers",
    name="Forecasted Price",
    line=dict(color="red", dash="dash")
))

fig.update_layout(
    title=f"{selected_stock} Stock Price Forecast ({steps}-Day Horizon)",
    xaxis_title="Date",
    yaxis_title="Price (INR)",
    hovermode="x unified",
    template="plotly_dark"
)

st.plotly_chart(fig, use_container_width=True)

# ------------------------------
# SUMMARY
# ------------------------------
st.markdown("### 💡 Quick Insights")
st.write(f"- **Latest Price:** ₹{st_data['Close'].iloc[-1]:.2f}")
st.write(f"- **Mean Daily Return:** {st_data['Returns'].mean() * 100:.2f}%")
st.write(f"- **Volatility (Std Dev of Returns):** {st_data['Returns'].std() * 100:.2f}%")

