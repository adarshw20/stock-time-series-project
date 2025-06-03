from statsmodels.tsa.arima.model import ARIMA

def train_arima(series, order=(5,1,0), steps=30):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast
