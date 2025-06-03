from prophet import Prophet
import pandas as pd

def train_prophet(df, periods=30):
    df_prophet = df.reset_index().rename(columns={"Date": "ds", "Close": "y"})
    model = Prophet()
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
