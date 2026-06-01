from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def run_sarimax(price, exog):

    train_size = int(len(price) * 0.8)

    train_y = price[:train_size]
    test_y = price[train_size:]

    train_x = exog[:train_size]
    test_x = exog[train_size:]

    model = SARIMAX(
        train_y,
        exog=train_x,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )

    fit = model.fit(disp=False)

    pred = fit.forecast(
        steps=len(test_y),
        exog=test_x
    )

    mae = mean_absolute_error(test_y, pred)
    rmse = np.sqrt(mean_squared_error(test_y, pred))

    print("\nSARIMAX")
    print("MAE :", mae)
    print("RMSE:", rmse)

    return mae, rmse