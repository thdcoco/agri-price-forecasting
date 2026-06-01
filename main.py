import pandas as pd

from src.arima_model import run_arima
from src.sarima_model import run_sarima


# =========================
# 데이터 로드
# =========================

df = pd.read_csv("data/kalimati_tarkari_dataset.csv")
df["Date"] = pd.to_datetime(df["Date"])

# =========================
# 타겟 선택 & 월별 리샘플링
# =========================

commodity = "Tomato Big(Nepali)"

data = df[df["Commodity"] == commodity].copy()
data = data.set_index("Date").sort_index()

# 일별 → 월별 평균, 빠진 달은 선형 보간
price = data["Average"].resample("MS").mean().interpolate("linear")

print(f"월별 데이터: {len(price)}개월 ({price.index[0].date()} ~ {price.index[-1].date()})")

# =========================
# 모델 실행
# =========================

print("\n=========================")
print("ARIMA RUN")
print("=========================")

run_arima(price)

print("\n=========================")
print("SARIMA RUN")
print("=========================")

run_sarima(price)