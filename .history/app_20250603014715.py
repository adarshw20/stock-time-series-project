import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model  # if you saved your LSTM model

# Load your preprocessed data (or download inside app)
@st.cache_data
def load_data():
    # Replace with your data loading logic
    df = pd.read_csv('your_stock_data.csv', parse_dates=['Date'])
    return df

# Prophet forecast function
def prophet_forecast(df):
    df_prophet = df[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast

# Example LSTM prediction function (simplified)
def lstm_forecast(df):
    # Load and preprocess data, then load your saved LSTM model and predict
    pass

def plot_stock(df):
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Close'], label='Actual Price')
    plt.title('Stock Closing Price')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(plt)

st.title("Stock Market Forecasting App")

df = load_data()
plot_stock(df)

st.subheader("Prophet Forecast")
forecast = prophet_forecast(df)
fig2 = Prophet.plot(forecast)
st.pyplot(fig2)

# Add LSTM and ARIMA similarly

st.write("Developed by Adarsh")
