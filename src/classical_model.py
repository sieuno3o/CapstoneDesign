from statsmodels.tsa.arima.model import ARIMA


def train_classical_model(series, order=(1, 0, 1)):
    model = ARIMA(series, order=order)
    fitted = model.fit()
    return fitted


def predict_classical_model(fitted_model, steps: int):
    return fitted_model.forecast(steps=steps)