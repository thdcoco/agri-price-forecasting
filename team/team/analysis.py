"""
한국 전력수요 SARIMA vs SARIMAX(M1/M2/M3) 분석
검증: 2023-01-01 ~ 2024-12-31 (4계절 × 2년)
"""
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from sklearn.metrics import mean_squared_error
import matplotlib.ticker as mticker

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

BASE         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR     = os.path.join(BASE, 'data_elec', 'data_elec')
FORECAST_DIR = os.path.join(BASE, 'data_elec', 'data_forcast')
OUT_DIR      = os.path.join(BASE, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

VALID_START = '2024-01-01'
VALID_END   = '2025-12-31'
ORDER          = (2, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 7)

# ════════════════════════════════════════════════════════════════
# 1. 전력수요 데이터 로드
# ════════════════════════════════════════════════════════════════
def load_elec_csv(path):
    df = pd.read_csv(path, encoding='cp949')
    df.columns = ['날짜'] + [str(i) for i in range(1, 25)]
    df['날짜'] = pd.to_datetime(df['날짜'].str.strip())
    hour_cols = [str(i) for i in range(1, 25)]
    df[hour_cols] = df[hour_cols].apply(pd.to_numeric, errors='coerce')
    df['load_mw'] = df[hour_cols].mean(axis=1)
    return df[['날짜', 'load_mw']]

elec_files = [
    '2013~2020 수요관리후 발전단 전력수요실적.csv',
    '2021년 1_12월 수요관리후 발전단 수요실적.csv',
    '시간별 전국 전력수요량_20231231.csv',
    '한국전력거래소_시간별 전국 전력수요량_20241231.csv',
    '한국전력거래소_시간별 전국 전력수요량_20251231.csv',
]

frames = [load_elec_csv(os.path.join(DATA_DIR, f)) for f in elec_files]
elec = pd.concat(frames).drop_duplicates('날짜').sort_values('날짜').reset_index(drop=True)
elec['load'] = elec['load_mw'] / 1000   # MW → GW
print(f"[전력수요] {elec['날짜'].min().date()} ~ {elec['날짜'].max().date()}  ({len(elec):,}일)")

# ════════════════════════════════════════════════════════════════
# 2. 기온 데이터 로드 (평균·최저·최고 모두)
# ════════════════════════════════════════════════════════════════
def load_temp_csv(path):
    df = pd.read_csv(path, encoding='cp949', skiprows=7)
    df.columns = ['날짜', '지점', '평균기온', '최저기온', '최고기온']
    df['날짜'] = pd.to_datetime(df['날짜'].str.strip())
    for col in ['평균기온', '최저기온', '최고기온']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df[['날짜', '평균기온', '최저기온', '최고기온']].dropna().reset_index(drop=True)

temp_hist = load_temp_csv(os.path.join(FORECAST_DIR, 'ta_20260602175659.csv'))
temp_2026 = load_temp_csv(os.path.join(FORECAST_DIR, 'ta_20260602175713.csv'))
print(f"[기온-과거] {temp_hist['날짜'].min().date()} ~ {temp_hist['날짜'].max().date()}  ({len(temp_hist):,}일)")
print(f"[기온-2026] {temp_2026['날짜'].min().date()} ~ {temp_2026['날짜'].max().date()}  ({len(temp_2026):,}일)")

# ════════════════════════════════════════════════════════════════
# 3. 병합 및 외생변수 생성
# ════════════════════════════════════════════════════════════════
def make_exog(df):
    """세 가지 외생변수 설계안 컬럼 생성"""
    # M1: 평균기온 기반
    df['CDD_avg'] = np.maximum(df['평균기온'] - 24, 0)
    df['HDD_avg'] = np.maximum(18 - df['평균기온'], 0)
    # M2: 최고·최저 분리
    df['CDD_max'] = np.maximum(df['최고기온'] - 24, 0)
    df['HDD_min'] = np.maximum(18 - df['최저기온'], 0)
    # M3: M2 + 열대야
    df['TROP']    = np.maximum(df['최저기온'] - 23, 0)
    return df

df = elec.merge(temp_hist, on='날짜', how='inner')
df = make_exog(df)
df = df.set_index('날짜').asfreq('D')

print(f"[병합결과] {df.index.min().date()} ~ {df.index.max().date()}  ({len(df):,}일)")
print(f"  결측치: {df[['load','평균기온']].isna().sum().to_dict()}")

interp_cols = ['load', '평균기온', '최저기온', '최고기온',
               'CDD_avg', 'HDD_avg', 'CDD_max', 'HDD_min', 'TROP']
df[interp_cols] = df[interp_cols].interpolate()

# 2026 외생변수
temp_2026 = make_exog(temp_2026.copy())
temp_2026 = temp_2026.set_index('날짜').asfreq('D')

# ════════════════════════════════════════════════════════════════
# 4. 정상성 검정 (ADF)
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("ADF 정상성 검정")
print("="*60)

def adf_test(series, label):
    result = adfuller(series.dropna(), autolag='AIC')
    stat, pval = result[0], result[1]
    print(f"  {label:<20} ADF={stat:8.3f}  p={pval:.4f}  {'정상' if pval < 0.05 else '비정상'}")

adf_test(df['load'],                   '원계열(load)')
adf_test(df['load'].diff().dropna(),   '1차차분')
adf_test(df['load'].diff(7).dropna(),  '계절차분(s=7)')

# ════════════════════════════════════════════════════════════════
# 5. 학습/검증 분할
# ════════════════════════════════════════════════════════════════
train = df.loc[:'2023-12-31']
valid = df.loc[VALID_START:VALID_END]
print(f"\n학습: {train.index.min().date()} ~ {train.index.max().date()}  ({len(train):,}일)")
print(f"검증: {valid.index.min().date()} ~ {valid.index.max().date()}  ({len(valid):,}일)")

# ════════════════════════════════════════════════════════════════
# 6. ACF/PACF 그래프
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
plot_acf( train['load'].diff().dropna(),        lags=42, ax=axes[0,0], title='ACF (1차차분)')
plot_pacf(train['load'].diff().dropna(),        lags=42, ax=axes[0,1], title='PACF (1차차분)')
plot_acf( train['load'].diff().diff(7).dropna(),lags=42, ax=axes[1,0], title='ACF (1차+계절차분)')
plot_pacf(train['load'].diff().diff(7).dropna(),lags=42, ax=axes[1,1], title='PACF (1차+계절차분)')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig0_acf_pacf.png'), dpi=150)
plt.close()
print("\n[저장] fig0_acf_pacf.png")

# ════════════════════════════════════════════════════════════════
# 7. 차수 결정 (ACF/PACF 기반)
# ════════════════════════════════════════════════════════════════
# 1차+계절 차분 후 ACF/PACF 해석:
#   PACF: lag 2에서 절단 → p=2
#   ACF:  lag 2에서 절단 → q=2
#   계절 PACF: lag 1(=7일)에서 절단 → P=1
#   계절 ACF:  lag 1(=7일)에서 절단 → Q=1
ORDER          = (2, 1, 2)
SEASONAL_ORDER = (1, 1, 1, 7)
print(f"\n차수 결정 (ACF/PACF 기반): order={ORDER}, seasonal_order={SEASONAL_ORDER}")

# ════════════════════════════════════════════════════════════════
# 8. 4개 모델 정의
# ════════════════════════════════════════════════════════════════
MODELS = {
    'SARIMA': [],
    'M1':     ['CDD_avg', 'HDD_avg'],
    'M2':     ['CDD_max', 'HDD_min'],
    'M3':     ['CDD_max', 'HDD_min', 'TROP'],
}
COLORS = {
    'SARIMA': 'royalblue',
    'M1':     'crimson',
    'M2':     'darkorange',
    'M3':     'green',
}

# ════════════════════════════════════════════════════════════════
# 8. 모델 적합 & 검증 예측
# ════════════════════════════════════════════════════════════════
results  = {}   # 적합 결과
preds    = {}   # 검증 예측값
perf     = {}   # RMSE, MAPE

def rmse(y, yhat): return np.sqrt(mean_squared_error(y, yhat))
def mape(y, yhat): return np.mean(np.abs((y - yhat) / y)) * 100

steps = len(valid)
actual = valid['load'].values

for name, cols in MODELS.items():
    print(f"\n모델 {name} 학습 중...")
    exog_train = train[cols] if cols else None
    exog_valid = valid[cols] if cols else None

    res = SARIMAX(train['load'],
                  exog=exog_train,
                  order=ORDER,
                  seasonal_order=SEASONAL_ORDER,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False, maxiter=200)
    results[name] = res
    print(f"  AIC = {res.aic:.2f}")

    fc = res.forecast(steps=steps, exog=exog_valid)
    preds[name] = fc.values
    perf[name]  = {'AIC': res.aic,
                   'RMSE': rmse(actual, fc.values),
                   'MAPE': mape(actual, fc.values)}

# ════════════════════════════════════════════════════════════════
# 9. 성능 비교 출력
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("모델 성능 비교 (검증: 2023~2024, 4계절×2년)")
print("="*60)
print(f"{'모델':<10} {'AIC':>10} {'RMSE(GW)':>12} {'MAPE(%)':>10}")
print("-"*44)
for name, p in perf.items():
    print(f"{name:<10} {p['AIC']:>10.2f} {p['RMSE']:>12.4f} {p['MAPE']:>10.2f}")

# ════════════════════════════════════════════════════════════════
# 10. 잔차 진단
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("잔차 진단 (Ljung-Box 검정)")
print("="*60)

def plot_diagnostics(res, name, color):
    resid = res.resid.dropna()

    # Ljung-Box 검정 (lag 10)
    lb = acorr_ljungbox(resid, lags=10, return_df=True)
    lb_pval_min = lb['lb_pvalue'].min()
    print(f"  [{name}] Ljung-Box p-value (최소, lag 1~10): {lb_pval_min:.4f} "
          f"  >> {'잔차 독립 OK' if lb_pval_min > 0.05 else '자기상관 잔존 FAIL'}")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'잔차 진단 — {name}', fontsize=13)

    # ① 잔차 시계열
    axes[0, 0].plot(resid.index, resid.values, color=color, lw=0.6)
    axes[0, 0].axhline(0, color='black', lw=0.8, ls='--')
    axes[0, 0].set_title('잔차 시계열')
    axes[0, 0].set_ylabel('잔차 (GW)')

    # ② 잔차 ACF
    plot_acf(resid, lags=40, ax=axes[0, 1], title='잔차 ACF', color=color)

    # ③ QQ 플롯
    stats.probplot(resid, dist='norm', plot=axes[1, 0])
    axes[1, 0].set_title('QQ 플롯 (정규성 확인)')
    axes[1, 0].get_lines()[0].set(color=color, markersize=2, alpha=0.5)
    axes[1, 0].get_lines()[1].set(color='red', lw=1.2)

    # ④ 잔차 히스토그램
    axes[1, 1].hist(resid, bins=60, color=color, alpha=0.7, edgecolor='white')
    xr = np.linspace(resid.min(), resid.max(), 200)
    axes[1, 1].plot(xr,
                    stats.norm.pdf(xr, resid.mean(), resid.std()) * len(resid) * (resid.max()-resid.min()) / 60,
                    color='red', lw=1.5, label='정규분포')
    axes[1, 1].set_title('잔차 분포')
    axes[1, 1].legend()

    plt.tight_layout()
    fname = os.path.join(OUT_DIR, f'fig6_diag_{name}.png')
    plt.savefig(fname, dpi=150)
    plt.close()
    print(f"  [저장] fig6_diag_{name}.png")

for name, res in results.items():
    plot_diagnostics(res, name, COLORS[name])

# ════════════════════════════════════════════════════════════════
# 11. 계절별 RMSE 분리
# ════════════════════════════════════════════════════════════════
SEASONS = {
    '봄(3~5월)':   [3, 4, 5],
    '여름(6~8월)': [6, 7, 8],
    '가을(9~11월)':[9, 10, 11],
    '겨울(12~2월)':[12, 1, 2],
}

print("\n" + "="*60)
print("계절별 RMSE 비교")
print("="*60)
print(f"{'계절':<14}", end='')
for name in MODELS:
    print(f"  {name:>10}", end='')
print()
print("-" * (14 + 12 * len(MODELS)))

for season_name, months in SEASONS.items():
    mask = valid.index.month.isin(months)
    act_s = valid.loc[mask, 'load'].values
    print(f"{season_name:<14}", end='')
    for name in MODELS:
        pred_s = np.array(preds[name])[mask]
        r = rmse(act_s, pred_s)
        print(f"  {r:>10.3f}", end='')
    print()

# ════════════════════════════════════════════════════════════════
# 12. 외생변수 계수 출력
# ════════════════════════════════════════════════════════════════
print("\n" + "="*60)
print("외생변수 계수 (기온 1도일당 수요 변화량)")
print("="*60)
for name, cols in MODELS.items():
    if not cols:
        continue
    print(f"\n  [{name}]")
    for col in cols:
        coef = results[name].params[col]
        pval = results[name].pvalues[col]
        sig  = '***' if pval < 0.001 else ('**' if pval < 0.01 else ('*' if pval < 0.05 else ''))
        print(f"    {col:<10} 계수={coef:+.4f} GW/도일   p={pval:.4f} {sig}")

# ════════════════════════════════════════════════════════════════
# 13. 전체 데이터(2013~2025)로 재학습 → 2026 예측
# ════════════════════════════════════════════════════════════════
print("\n전체 데이터로 재학습 중...")
full = df.loc[:'2025-12-31']
full_results = {}
full_preds   = {}
full_ci      = {}

for name, cols in MODELS.items():
    exog_full = full[cols] if cols else None
    exog_2026 = temp_2026[cols] if cols else None

    res_full = SARIMAX(full['load'],
                       exog=exog_full,
                       order=ORDER,
                       seasonal_order=SEASONAL_ORDER,
                       enforce_stationarity=False,
                       enforce_invertibility=False).fit(disp=False, maxiter=200)
    full_results[name] = res_full

    fc = res_full.get_forecast(steps=len(temp_2026), exog=exog_2026)
    full_preds[name] = fc.predicted_mean
    full_preds[name].index = temp_2026.index
    if name != 'SARIMA':
        ci = fc.conf_int(alpha=0.05)
        ci.index = temp_2026.index
        full_ci[name] = ci

print(f"  2026 예측 완료: {temp_2026.index.min().date()} ~ {temp_2026.index.max().date()}")

# 2026 월별 평균 예측값 표
print("\n" + "="*60)
print("2026 상반기 월별 평균 전력수요 예측 (GW)")
print("="*60)
monthly = pd.DataFrame({'SARIMA': full_preds['SARIMA'], 'SARIMAX-M1': full_preds['M1']})
monthly['월'] = monthly.index.month
summary = monthly.groupby('월')[['SARIMA', 'SARIMAX-M1']].mean().round(2)
summary.index = [f"{m}월" for m in summary.index]
print(summary.to_string())

# ════════════════════════════════════════════════════════════════
# 14. 그래프
# ════════════════════════════════════════════════════════════════

# ① 전체 시계열
fig, ax = plt.subplots(figsize=(16, 4))
ax.plot(df.index, df['load'], color='steelblue', lw=0.7, label='실측 (GW)')
ax.set_title('한국 전국 전력수요 일평균 (2013~2025)', fontsize=13)
ax.set_ylabel('GW')
ax.legend()
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig1_timeseries.png'), dpi=150)
plt.close()
print("[저장] fig1_timeseries.png")

# ② 검증구간 4모델 비교
fig, ax = plt.subplots(figsize=(16, 5))
ax.plot(valid.index, actual, color='black', lw=1.5, label='실측', zorder=5)
for name, pred in preds.items():
    ls = '--' if name == 'SARIMA' else '-'
    ax.plot(valid.index, pred, color=COLORS[name], lw=1.1, ls=ls,
            label=f"{name}  RMSE={perf[name]['RMSE']:.3f}  MAPE={perf[name]['MAPE']:.1f}%")
ax.set_title(f'검증구간 4모델 비교 ({VALID_START} ~ {VALID_END}, 4계절×2년)', fontsize=13)
ax.set_ylabel('GW')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig2_validation.png'), dpi=150)
plt.close()
print("[저장] fig2_validation.png")

# ③ RMSE/MAPE 막대 비교
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
names = list(perf.keys())
rmse_vals = [perf[n]['RMSE'] for n in names]
mape_vals = [perf[n]['MAPE'] for n in names]
colors    = [COLORS[n] for n in names]

axes[0].bar(names, rmse_vals, color=colors, edgecolor='white')
axes[0].set_title('RMSE 비교 (GW)')
axes[0].set_ylabel('GW')
for i, v in enumerate(rmse_vals):
    axes[0].text(i, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)

axes[1].bar(names, mape_vals, color=colors, edgecolor='white')
axes[1].set_title('MAPE 비교 (%)')
axes[1].set_ylabel('%')
for i, v in enumerate(mape_vals):
    axes[1].text(i, v + 0.1, f'{v:.1f}%', ha='center', fontsize=9)

plt.suptitle('모델 성능 비교 (검증: 2023~2024)', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig3_bar_compare.png'), dpi=150)
plt.close()
print("[저장] fig3_bar_compare.png")

# ④ 외생변수 계수 비교
coef_data = {}
for name, cols in MODELS.items():
    if not cols:
        continue
    for col in cols:
        coef_data.setdefault(col, {})[name] = results[name].params[col]

fig, ax = plt.subplots(figsize=(10, 4))
x = np.arange(len(coef_data))
model_names = [n for n in MODELS if n != 'SARIMA']
width = 0.25
for i, mname in enumerate(model_names):
    vals = [coef_data[col].get(mname, 0) for col in coef_data]
    ax.bar(x + i * width, vals, width, label=mname, color=COLORS[mname], edgecolor='white')
ax.set_xticks(x + width)
ax.set_xticklabels(list(coef_data.keys()))
ax.set_title('외생변수 계수 비교 (GW/도일)')
ax.set_ylabel('계수')
ax.legend()
ax.axhline(0, color='black', lw=0.8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig4_coef_compare.png'), dpi=150)
plt.close()
print("[저장] fig4_coef_compare.png")

# ⑤ 2026 예측 (SARIMA vs 최적 SARIMAX)
best = min((n for n in perf if n != 'SARIMA'), key=lambda n: perf[n]['RMSE'])
fig, ax = plt.subplots(figsize=(14, 5))
recent = df.loc['2025-06-01':]
ax.plot(recent.index, recent['load'], color='black', lw=1.2, label='실측(2025 하반기)')
ax.plot(full_preds['SARIMA'].index, full_preds['SARIMA'].values,
        color=COLORS['SARIMA'], lw=1.5, ls='--', label='SARIMA 예측')
ax.plot(full_preds[best].index, full_preds[best].values,
        color=COLORS[best], lw=1.5, label=f'SARIMAX-{best} 예측 (최우수)')
ax.fill_between(full_ci[best].index,
                full_ci[best].iloc[:, 0],
                full_ci[best].iloc[:, 1],
                color=COLORS[best], alpha=0.15, label='95% CI')
ax.set_title(f'2026년 상반기 전력수요 예측 (SARIMA vs SARIMAX-{best})', fontsize=13)
ax.set_ylabel('GW')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'fig5_forecast_2026.png'), dpi=150)
plt.close()
print("[저장] fig5_forecast_2026.png")

print("\n분석 완료!")
