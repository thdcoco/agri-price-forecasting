from statsmodels.tsa.api import VAR
import numpy as np
from sklearn.metrics import mean_squared_error


def run_var(df_multi):

    train_size = int(len(df_multi) * 0.8)

    train = df_multi[:train_size]
    test = df_multi[train_size:]

    model = VAR(train)
    fit = model.fit(2)

    lag = fit.k_ar
    forecast = fit.forecast(train.values[-lag:], len(test))

    rmse = np.sqrt(mean_squared_error(test, forecast))

    print("\nVAR")
    print("RMSE:", rmse)

    return rmse