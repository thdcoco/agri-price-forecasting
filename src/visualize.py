import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Windows 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def load_monthly_price():
    df = pd.read_csv("data/kalimati_tarkari_dataset.csv")
    df["Date"] = pd.to_datetime(df["Date"])
    data = df[df["Commodity"] == "Tomato Big(Nepali)"].copy()
    data = data.set_index("Date").sort_index()
    price = data["Average"].resample("MS").mean().interpolate("linear")
    return price


def fit_models(price):
    train_size = int(len(price) * 0.8)
    train = price[:train_size]
    test  = price[train_size:]

    arima_fit  = ARIMA(train, order=(1,1,1)).fit()
    sarima_fit = SARIMAX(train, order=(1,1,1), seasonal_order=(1,1,1,12)).fit(disp=False)

    arima_pred  = arima_fit.forecast(len(test))
    sarima_pred = sarima_fit.forecast(len(test))

    arima_pred.index  = test.index
    sarima_pred.index = test.index

    return train, test, arima_pred, sarima_pred


def plot_seasonality(price, ax):
    monthly_avg = price.groupby(price.index.month).mean()
    monthly_std = price.groupby(price.index.month).std()

    month_labels = ["1월","2월","3월","4월","5월","6월",
                    "7월","8월","9월","10월","11월","12월"]

    ax.fill_between(range(1, 13),
                    monthly_avg - monthly_std,
                    monthly_avg + monthly_std,
                    alpha=0.2, color="steelblue", label="±1 표준편차")
    ax.plot(range(1, 13), monthly_avg,
            marker="o", color="steelblue", linewidth=2, label="월별 평균가")

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels)
    ax.set_title("월별 계절성 패턴 (Tomato Big, 2013–2021)", fontsize=13, fontweight="bold")
    ax.set_ylabel("평균 가격 (루피/kg)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_predictions(price, train, test, arima_pred, sarima_pred, ax):
    ax.plot(train.index, train.values,
            color="gray", linewidth=1.5, label="학습 데이터 (Train)")
    ax.plot(test.index, test.values,
            color="black", linewidth=2, label="실제값 (Test)")
    ax.plot(test.index, arima_pred.values,
            color="tomato", linewidth=1.8, linestyle="--", label="ARIMA 예측")
    ax.plot(test.index, sarima_pred.values,
            color="steelblue", linewidth=1.8, linestyle="--", label="SARIMA 예측")

    ax.axvline(x=test.index[0], color="black", linestyle=":", linewidth=1.2)
    ax.text(test.index[0], ax.get_ylim()[0], " ← Train | Test →",
            fontsize=8, color="black", va="bottom")

    ax.set_title("실제값 vs 예측값 비교", fontsize=13, fontweight="bold")
    ax.set_ylabel("평균 가격 (루피/kg)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_performance(test, arima_pred, sarima_pred, ax):
    arima_mae  = mean_absolute_error(test, arima_pred)
    arima_rmse = np.sqrt(mean_squared_error(test, arima_pred))
    sarima_mae  = mean_absolute_error(test, sarima_pred)
    sarima_rmse = np.sqrt(mean_squared_error(test, sarima_pred))

    labels    = ["ARIMA", "SARIMA"]
    mae_vals  = [arima_mae, sarima_mae]
    rmse_vals = [arima_rmse, sarima_rmse]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax.bar(x - width/2, mae_vals,  width, label="MAE",  color=["tomato", "steelblue"],   alpha=0.85)
    bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE", color=["tomato", "steelblue"],   alpha=0.5)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_title("ARIMA vs SARIMA 성능 비교 (MAE / RMSE)", fontsize=13, fontweight="bold")
    ax.set_ylabel("오차 (루피/kg)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def plot_before_after(test, arima_pred, sarima_pred, ax):
    arima_mae_new   = mean_absolute_error(test, arima_pred)
    arima_rmse_new  = np.sqrt(mean_squared_error(test, arima_pred))
    sarima_mae_new  = mean_absolute_error(test, sarima_pred)
    sarima_rmse_new = np.sqrt(mean_squared_error(test, sarima_pred))

    # 기존 일별 결과 (하드코딩)
    old = {"ARIMA\n(일별)":  (19.34, 22.48),
           "SARIMA\n(일별)": (23.56, 27.35)}
    new = {"ARIMA\n(월별)":  (arima_mae_new,  arima_rmse_new),
           "SARIMA\n(월별)": (sarima_mae_new, sarima_rmse_new)}

    all_models = {**old, **new}
    labels    = list(all_models.keys())
    mae_vals  = [v[0] for v in all_models.values()]
    rmse_vals = [v[1] for v in all_models.values()]

    x = np.arange(len(labels))
    width = 0.35
    mae_colors  = ["#d9534f", "#d9534f", "#5bc0de", "#5bc0de"]
    rmse_colors = ["#c9302c", "#c9302c", "#31b0d5", "#31b0d5"]

    bars1 = ax.bar(x - width/2, mae_vals,  width, label="MAE",  color=mae_colors,  alpha=0.85)
    bars2 = ax.bar(x + width/2, rmse_vals, width, label="RMSE", color=rmse_colors, alpha=0.5)

    for bar in bars1 + bars2:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.2,
                f"{bar.get_height():.2f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.axvline(x=1.5, color="black", linestyle=":", linewidth=1.2)
    ymax = ax.get_ylim()[1]
    ax.text(0.75, ymax * 0.93, "기존 (일별, m=12 오류)", ha="center", fontsize=9, color="#d9534f")
    ax.text(2.25, ymax * 0.93, "수정 후 (월별, m=12)",   ha="center", fontsize=9, color="#5bc0de")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_title("기존 vs 월별 리샘플링 성능 비교 (MAE / RMSE)", fontsize=13, fontweight="bold")
    ax.set_ylabel("오차 (루피/kg)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)


def run_visualization():
    print("데이터 로딩 중...")
    price = load_monthly_price()

    print("모델 학습 중...")
    train, test, arima_pred, sarima_pred = fit_models(price)

    fig, axes = plt.subplots(4, 1, figsize=(12, 20))
    fig.suptitle("Tomato Big(Nepali) 가격 예측 — ARIMA / SARIMA",
                 fontsize=15, fontweight="bold", y=0.99)

    plot_seasonality(price, axes[0])
    plot_predictions(price, train, test, arima_pred, sarima_pred, axes[1])
    plot_performance(test, arima_pred, sarima_pred, axes[2])
    plot_before_after(test, arima_pred, sarima_pred, axes[3])

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig("report_figures.png", dpi=150, bbox_inches="tight")
    print("저장 완료: report_figures.png")
    plt.show()


if __name__ == "__main__":
    run_visualization()
