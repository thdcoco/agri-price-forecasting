from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def run_arima(price):

    train_size = int(len(price) * 0.8)

    train = price[:train_size]
    test = price[train_size:]

    model = ARIMA(train, order=(1,1,1))
    fit = model.fit()

    pred = fit.forecast(len(test))

    mae = mean_absolute_error(test, pred)
    rmse = np.sqrt(mean_squared_error(test, pred))

    print("\nARIMA")
    print("MAE :", mae)
    print("RMSE:", rmse)

    return mae, rmse