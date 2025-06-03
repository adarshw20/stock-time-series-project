import streamlit as st
from data_preprocessing import load_stock_data
from model_arima import train_arima
from model_prophet import train_prophet
import matplotlib.pyplot as plt

st.title("📈 Stock Market Forecasting Dashboard")

ticker = st.text_input("Enter Stock Ticker", "AAPL")
df = load_stock_data(ticker)

st.subheader("📊 Historical Closing Price")
st.line_chart(df['Close'])

model_choice = st.selectbox("Choose Forecasting Model", ["ARIMA", "Prophet"])

if model_choice == "ARIMA":
    forecast = train_arima(df['Close'])
    st.subheader("📉 ARIMA Forecast")
    fig, ax = plt.subplots()
    df['Close'].iloc[-60:].plot(ax=ax, label="Historical")
    forecast.plot(ax=ax, label="Forecast")
    ax.legend()
    st.pyplot(fig)

elif model_choice == "Prophet":
    forecast_df = train_prophet(df)
    st.subheader("📉 Prophet Forecast")
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Close'], label='Historical')
    ax.plot(forecast_df['ds'], forecast_df['yhat'], label='Forecast')
    ax.fill_between(forecast_df['ds'], forecast_df['yhat_lower'], forecast_df['yhat_upper'], alpha=0.3)
    ax.legend()
    st.pyplot(fig)
