from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def run_sarima(price):

    train_size = int(len(price) * 0.8)

    train = price[:train_size]
    test = price[train_size:]

    model = SARIMAX(
        train,
        order=(1,1,1),
        seasonal_order=(1,1,1,12)
    )

    fit = model.fit()

    pred = fit.forecast(len(test))

    mae = mean_absolute_error(test, pred)

    rmse = np.sqrt(
        mean_squared_error(
            test,
            pred
        )
    )

    print("\nSARIMA")
    print("MAE :", mae)
    print("RMSE:", rmse)


if __name__ == "__main__":

    import pandas as pd

    df = pd.read_csv(
        "data/kalimati_tarkari_dataset.csv"
    )

    df["Date"] = pd.to_datetime(
        df["Date"]
    )

    tomato = df[
        df["Commodity"] == "Tomato Big(Nepali)"
    ].copy()

    tomato = tomato.sort_values(
        "Date"
    )

    price = tomato["Average"]

    run_sarima(price)