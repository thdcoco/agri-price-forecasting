import pandas as pd

from src.sarima_model import run_sarima
from src.sarimax_model import run_sarimax


# =========================
# 데이터 로드
# =========================

df = pd.read_csv("data/kalimati_tarkari_dataset.csv")
weather_df = pd.read_csv("data/Nepal_Terai_2013-2021.csv")

df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 타겟 선택
# =========================

commodity = "Tomato Big(Nepali)"

data = df[df["Commodity"] == commodity].copy()
data = data.sort_values("Date")

price = data["Average"].reset_index(drop=True)

# =========================
# 외생변수 (기상)
# =========================

exog = weather_df[["T2M", "PRECTOTCORR"]].reset_index(drop=True)

# 길이 맞추기
min_len = min(len(price), len(exog))

price = price[:min_len]
exog = exog[:min_len]

# =========================
# 모델 실행
# =========================

print("\n=========================")
print("SARIMA RUN")
print("=========================")

run_sarima(price)

print("\n=========================")
print("SARIMAX RUN")
print("=========================")

run_sarimax(price, exog)