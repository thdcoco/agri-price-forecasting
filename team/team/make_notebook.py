"""
analysis_elec.ipynb 생성 스크립트
SARIMA vs SARIMAX (외생변수 설계 3종 비교)
"""
import json, os

def code(src, cid):
    return {"cell_type": "code", "execution_count": None,
            "id": cid, "metadata": {}, "outputs": [],
            "source": src}

def md(src, cid):
    return {"cell_type": "markdown", "id": cid,
            "metadata": {}, "source": src}

cells = []

# ─── 0. 제목 ────────────────────────────────────────────────────
cells.append(md(
    "# 한국 전력수요 SARIMA vs SARIMAX 분석\n"
    "## 외생변수 설계 3가지 비교\n"
    "\n"
    "| 모델 | 외생변수 | 근거 |\n"
    "|------|---------|------|\n"
    "| SARIMA | 없음 | 기온 효과 미반영 |\n"
    "| SARIMAX-M1 | CDD_avg, HDD_avg | 평균기온 기반 (베이스라인) |\n"
    "| SARIMAX-M2 | CDD_max, HDD_min | 최고·최저 분리 (물리적 근거) |\n"
    "| SARIMAX-M3 | CDD_max, HDD_min, TROP | M2 + 열대야 추가 |\n"
    "\n"
    "> **핵심**: 여름 냉방은 낮 더위(최고기온), 겨울 난방은 새벽 추위(최저기온)가 주도\n"
    "> → 평균기온 1개보다 최고·최저 분리가 물리적으로 더 타당",
    "md-title"
))

# ─── 1. 임포트 (CH08 패턴) ────────────────────────────────────
cells.append(code(
    "from sklearn.metrics import mean_squared_error\n"
    "from statsmodels.graphics.tsaplots import plot_acf, plot_pacf\n"
    "from statsmodels.tsa.seasonal import STL\n"
    "from statsmodels.stats.diagnostic import acorr_ljungbox\n"
    "from statsmodels.tsa.statespace.sarimax import SARIMAX\n"
    "from statsmodels.tsa.stattools import adfuller\n"
    "from tqdm import tqdm_notebook\n"
    "from itertools import product\n"
    "from typing import Union\n"
    "\n"
    "import matplotlib.pyplot as plt\n"
    "import matplotlib.gridspec as gridspec\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import glob, os\n"
    "\n"
    "import warnings\n"
    "warnings.filterwarnings('ignore')\n"
    "\n"
    "%matplotlib inline\n"
    "plt.rcParams['font.family'] = 'Malgun Gothic'\n"
    "plt.rcParams['axes.unicode_minus'] = False",
    "imports"
))

# ─── 2. 전력수요 로드 ─────────────────────────────────────────
cells.append(md("## 1. 데이터 로드 & 전처리", "md-data"))

cells.append(code(
    "DATA_DIR     = 'data_elec/data_elec'\n"
    "FORECAST_DIR = 'data_elec/data_forcast'\n"
    "\n"
    "def load_elec_csv(path):\n"
    "    df = pd.read_csv(path, encoding='cp949')\n"
    "    df.columns = ['날짜'] + [str(i) for i in range(1, 25)]\n"
    "    df['날짜'] = pd.to_datetime(df['날짜'].str.strip())\n"
    "    h = [str(i) for i in range(1, 25)]\n"
    "    df[h] = df[h].apply(pd.to_numeric, errors='coerce')\n"
    "    df['load_mw'] = df[h].mean(axis=1)\n"
    "    return df[['날짜', 'load_mw']]\n"
    "\n"
    "elec = (pd.concat([load_elec_csv(f) for f in sorted(glob.glob(os.path.join(DATA_DIR,'*.csv')))])\n"
    "          .drop_duplicates('날짜').sort_values('날짜').reset_index(drop=True))\n"
    "elec['load'] = elec['load_mw'] / 1000\n"
    "print(f'전력수요: {elec[\"날짜\"].min().date()} ~ {elec[\"날짜\"].max().date()}  ({len(elec):,}일)')\n"
    "elec.head()",
    "load-elec"
))

# ─── 3. 기온 로드 (평균·최저·최고 모두) ──────────────────────
cells.append(code(
    "def load_temp_csv(path):\n"
    "    df = pd.read_csv(path, encoding='cp949', skiprows=6)\n"
    "    df.columns = ['날짜', '지점', '평균기온', '최저기온', '최고기온']\n"
    "    df['날짜'] = pd.to_datetime(df['날짜'].str.strip())\n"
    "    for col in ['평균기온', '최저기온', '최고기온']:\n"
    "        df[col] = pd.to_numeric(df[col], errors='coerce')\n"
    "    return df[['날짜', '평균기온', '최저기온', '최고기온']].dropna(subset=['평균기온']).reset_index(drop=True)\n"
    "\n"
    "temp_hist = load_temp_csv(os.path.join(FORECAST_DIR, 'ta_20260602175659.csv'))\n"
    "temp_2026 = load_temp_csv(os.path.join(FORECAST_DIR, 'ta_20260602175713.csv'))\n"
    "\n"
    "print(f'기온(과거): {temp_hist[\"날짜\"].min().date()} ~ {temp_hist[\"날짜\"].max().date()}')\n"
    "print(f'기온(2026): {temp_2026[\"날짜\"].min().date()} ~ {temp_2026[\"날짜\"].max().date()}')\n"
    "temp_hist.head()",
    "load-temp"
))

# ─── 4. 병합 & 외생변수 3종 생성 ─────────────────────────────
cells.append(code(
    "df = elec.merge(temp_hist, on='날짜', how='inner')\n"
    "\n"
    "# ── M1: 평균기온 기반 (베이스라인) ──\n"
    "df['CDD_avg'] = np.maximum(df['평균기온'] - 24, 0)\n"
    "df['HDD_avg'] = np.maximum(18 - df['평균기온'],  0)\n"
    "\n"
    "# ── M2: 최고·최저 분리 (물리적 근거) ──\n"
    "# 여름 냉방 = 낮 더위 → 최고기온\n"
    "df['CDD_max'] = np.maximum(df['최고기온'] - 24, 0)\n"
    "# 겨울 난방 = 새벽 추위 → 최저기온\n"
    "df['HDD_min'] = np.maximum(18 - df['최저기온'],  0)\n"
    "\n"
    "# ── M3: M2 + 열대야 (밤에도 냉방 지속) ──\n"
    "df['TROP']    = np.maximum(df['최저기온'] - 23, 0)\n"
    "\n"
    "df = df.set_index('날짜').asfreq('D')\n"
    "df = df.interpolate()\n"
    "\n"
    "# 2026 외생변수\n"
    "t26 = temp_2026.copy()\n"
    "t26['CDD_avg'] = np.maximum(t26['평균기온'] - 24, 0)\n"
    "t26['HDD_avg'] = np.maximum(18 - t26['평균기온'],  0)\n"
    "t26['CDD_max'] = np.maximum(t26['최고기온'] - 24, 0)\n"
    "t26['HDD_min'] = np.maximum(18 - t26['최저기온'],  0)\n"
    "t26['TROP']    = np.maximum(t26['최저기온'] - 23, 0)\n"
    "t26 = t26.set_index('날짜').asfreq('D')\n"
    "\n"
    "exog_2026 = {'M1': t26[['CDD_avg','HDD_avg']],\n"
    "             'M2': t26[['CDD_max','HDD_min']],\n"
    "             'M3': t26[['CDD_max','HDD_min','TROP']]}\n"
    "\n"
    "print(f'병합: {df.index.min().date()} ~ {df.index.max().date()}  ({len(df):,}일)')\n"
    "df[['load','CDD_avg','CDD_max','HDD_avg','HDD_min','TROP']].describe().round(2)",
    "merge"
))

# ─── 5. EDA ──────────────────────────────────────────────────
cells.append(md("## 2. 탐색적 분석", "md-eda"))

cells.append(code(
    "fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)\n"
    "\n"
    "axes[0].plot(df.index, df['load'], color='steelblue', lw=0.7)\n"
    "axes[0].set_ylabel('전력수요 (GW)')\n"
    "axes[0].set_title('한국 전국 전력수요 일평균 (2013~2025)')\n"
    "\n"
    "axes[1].plot(df.index, df['평균기온'], color='gray',   lw=0.6, label='평균기온')\n"
    "axes[1].plot(df.index, df['최고기온'], color='tomato', lw=0.5, alpha=0.6, label='최고기온')\n"
    "axes[1].plot(df.index, df['최저기온'], color='steelblue', lw=0.5, alpha=0.6, label='최저기온')\n"
    "axes[1].set_ylabel('기온 (℃)')\n"
    "axes[1].set_title('전국 기온 (평균·최고·최저)')\n"
    "axes[1].legend(loc='upper right', fontsize=8)\n"
    "\n"
    "axes[2].plot(df.index, df['CDD_avg'], color='tomato',    lw=0.7, label='CDD_avg (M1)')\n"
    "axes[2].plot(df.index, df['CDD_max'], color='darkred',   lw=0.7, ls='--', label='CDD_max (M2)')\n"
    "axes[2].plot(df.index, df['HDD_min'], color='navy',      lw=0.7, label='HDD_min (M2)')\n"
    "axes[2].plot(df.index, df['TROP'],    color='orange',    lw=0.7, ls=':', label='TROP (M3)')\n"
    "axes[2].set_ylabel('도일 값')\n"
    "axes[2].set_title('외생변수 시계열 비교')\n"
    "axes[2].legend(loc='upper right', fontsize=8)\n"
    "\n"
    "for ax in axes:\n"
    "    ax.spines['top'].set_alpha(0)\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig1_timeseries.png', dpi=150)\n"
    "plt.show()",
    "fig1"
))

# ─── 6. Fig2: STL ─────────────────────────────────────────────
cells.append(code(
    "decomp = STL(df['load'], period=365).fit()\n"
    "fig, axes = plt.subplots(4, 1, sharex=True, figsize=(14, 10))\n"
    "for ax, lab, s in zip(axes,\n"
    "                      ['Observed','Trend','Seasonal','Residuals'],\n"
    "                      [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid]):\n"
    "    ax.plot(s, lw=0.7)\n"
    "    ax.set_ylabel(lab)\n"
    "    ax.spines['top'].set_alpha(0)\n"
    "fig.suptitle('STL 분해 — 전력수요 (period=365)', fontsize=13)\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig2_stl.png', dpi=150)\n"
    "plt.show()",
    "fig2"
))

# ─── 7. Fig3: 산점도 — 평균 vs 최고·최저 비교 ────────────────
cells.append(code(
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))\n"
    "\n"
    "sc1 = ax1.scatter(df['평균기온'], df['load'],\n"
    "                  s=2, alpha=0.3, c=df.index.month, cmap='RdYlBu_r')\n"
    "ax1.set_xlabel('전국 평균기온 (℃)')\n"
    "ax1.set_ylabel('전력수요 (GW)')\n"
    "ax1.set_title('M1 기준: 평균기온 vs 전력수요')\n"
    "plt.colorbar(sc1, ax=ax1, label='월')\n"
    "\n"
    "ax2.scatter(df['최고기온'], df['load'],\n"
    "            s=2, alpha=0.2, color='tomato', label='최고기온')\n"
    "ax2.scatter(df['최저기온'], df['load'],\n"
    "            s=2, alpha=0.2, color='steelblue', label='최저기온')\n"
    "ax2.set_xlabel('기온 (℃)')\n"
    "ax2.set_ylabel('전력수요 (GW)')\n"
    "ax2.set_title('M2 기준: 최고·최저기온 vs 전력수요')\n"
    "ax2.legend(markerscale=5)\n"
    "\n"
    "plt.suptitle('기온 측정값에 따른 수요 관계 비교', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig3_scatter.png', dpi=150)\n"
    "plt.show()",
    "fig3"
))

# ─── 8. ADF 정상성 검정 (CH08 패턴) ──────────────────────────
cells.append(md("## 3. 정상성 검정 (ADF)", "md-adf"))

cells.append(code(
    "print('=' * 58)\n"
    "print('ADF 정상성 검정 결과')\n"
    "print('=' * 58)\n"
    "print(f'{\"구분\":<24} {\"ADF통계량\":>10} {\"p-value\":>10} {\"판정\":>8}')\n"
    "print('-' * 58)\n"
    "\n"
    "def adf_row(series, label):\n"
    "    r = adfuller(series.dropna(), autolag='AIC')\n"
    "    verdict = '정상 ✓' if r[1] < 0.05 else '비정상 ✗'\n"
    "    print(f'{label:<24} {r[0]:>10.3f} {r[1]:>10.4f} {verdict:>8}')\n"
    "    return r[1]\n"
    "\n"
    "adf_row(df['load'],                         '원계열')\n"
    "adf_row(df['load'].diff().dropna(),          '1차차분 (d=1)')\n"
    "adf_row(df['load'].diff(7).dropna(),         '주간계절차분 (s=7)')\n"
    "p = adf_row(df['load'].diff().diff(7).dropna(), '1차 + 계절차분')\n"
    "print(f'\\n→ d=1, D=1, s=7 사용')",
    "adf"
))

# ─── 9. Fig4: ACF/PACF (CH08 패턴) ───────────────────────────
cells.append(code(
    "fig, axes = plt.subplots(2, 2, figsize=(14, 8))\n"
    "plot_acf(df['load'].diff().dropna(),\n"
    "         lags=42, ax=axes[0,0], title='ACF — 1차차분')\n"
    "plot_pacf(df['load'].diff().dropna(),\n"
    "          lags=42, ax=axes[0,1], title='PACF — 1차차분')\n"
    "plot_acf(df['load'].diff().diff(7).dropna(),\n"
    "         lags=42, ax=axes[1,0], title='ACF — 1차+계절차분 (s=7)')\n"
    "plot_pacf(df['load'].diff().diff(7).dropna(),\n"
    "          lags=42, ax=axes[1,1], title='PACF — 1차+계절차분 (s=7)')\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig4_acf_pacf.png', dpi=150)\n"
    "plt.show()",
    "fig4"
))

# ─── 10. 학습/검증 분할 ──────────────────────────────────────
cells.append(md("## 4. 학습 / 검증 분할", "md-split"))

cells.append(code(
    "VALID_START = '2024-06-01'\n"
    "VALID_END   = '2024-08-31'\n"
    "\n"
    "train = df.loc[:'2024-05-31']\n"
    "valid = df.loc[VALID_START:VALID_END]\n"
    "\n"
    "EXOG = {\n"
    "    'M1': ['CDD_avg', 'HDD_avg'],\n"
    "    'M2': ['CDD_max', 'HDD_min'],\n"
    "    'M3': ['CDD_max', 'HDD_min', 'TROP'],\n"
    "}\n"
    "\n"
    "print(f'학습: {train.index.min().date()} ~ {train.index.max().date()}  ({len(train):,}일)')\n"
    "print(f'검증: {valid.index.min().date()} ~ {valid.index.max().date()}  ({len(valid):,}일)')\n"
    "\n"
    "fig, ax = plt.subplots(figsize=(14, 4))\n"
    "ax.plot(df.index, df['load'], color='steelblue', lw=0.6)\n"
    "ax.axvspan(valid.index[0], valid.index[-1],\n"
    "           color='#808080', alpha=0.25, label='검증구간 (여름 피크)')\n"
    "ax.set_ylabel('전력수요 (GW)')\n"
    "ax.set_title('학습 / 검증 분할')\n"
    "ax.legend()\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig5_split.png', dpi=150)\n"
    "plt.show()",
    "split"
))

# ─── 11. 차수 탐색 (CH08 optimize_SARIMA) ────────────────────
cells.append(md("## 5. 모델 차수 탐색 (SARIMA Grid Search)", "md-order"))

cells.append(code(
    "# CH08 optimize_SARIMA 패턴 — 모든 SARIMAX 모델이 동일 차수 사용\n"
    "def optimize_SARIMA(endog: Union[pd.Series, list],\n"
    "                    order_list: list, d: int, D: int, s: int) -> pd.DataFrame:\n"
    "    results = []\n"
    "    for order in tqdm_notebook(order_list):\n"
    "        try:\n"
    "            model = SARIMAX(endog,\n"
    "                            order=(order[0], d, order[1]),\n"
    "                            seasonal_order=(order[2], D, order[3], s),\n"
    "                            simple_differencing=False).fit(disp=False)\n"
    "            results.append([order, model.aic])\n"
    "        except:\n"
    "            continue\n"
    "    result_df = pd.DataFrame(results, columns=['(p,q,P,Q)', 'AIC'])\n"
    "    return result_df.sort_values('AIC').reset_index(drop=True)\n"
    "\n"
    "d=1; D=1; s=7\n"
    "order_list = list(product(range(0,3), range(0,3), range(0,2), range(0,2)))\n"
    "print(f'탐색 조합: {len(order_list)}개')\n"
    "\n"
    "SARIMA_result = optimize_SARIMA(train['load'], order_list, d, D, s)\n"
    "SARIMA_result.head(10)",
    "opt-sarima"
))

cells.append(code(
    "p, q, P, Q = SARIMA_result.iloc[0]['(p,q,P,Q)']\n"
    "print(f'선택 차수: SARIMA({p},{d},{q})({P},{D},{Q},{s})')\n"
    "print(f'이 차수를 SARIMA + SARIMAX-M1/M2/M3 모두에 동일하게 적용합니다.')",
    "best-order"
))

# ─── 12. 4개 모델 적합 ────────────────────────────────────────
cells.append(md("## 6. 모델 적합 (SARIMA + SARIMAX M1/M2/M3)", "md-fit"))

cells.append(code(
    "def fit_model(endog, exog_cols, p, d, q, P, D, Q, s, label):\n"
    "    exog = endog.to_frame().join(train[exog_cols])[exog_cols] if exog_cols else None\n"
    "    model = SARIMAX(endog, exog,\n"
    "                    order=(p, d, q),\n"
    "                    seasonal_order=(P, D, Q, s),\n"
    "                    simple_differencing=False)\n"
    "    res = model.fit(disp=False)\n"
    "    print(f'  {label:<14} AIC={res.aic:.2f}')\n"
    "    return res\n"
    "\n"
    "print('모델 적합 중...')\n"
    "res_sarima = fit_model(train['load'], [],                 p,d,q,P,D,Q,s, 'SARIMA')\n"
    "res_m1     = fit_model(train['load'], EXOG['M1'],         p,d,q,P,D,Q,s, 'SARIMAX-M1')\n"
    "res_m2     = fit_model(train['load'], EXOG['M2'],         p,d,q,P,D,Q,s, 'SARIMAX-M2')\n"
    "res_m3     = fit_model(train['load'], EXOG['M3'],         p,d,q,P,D,Q,s, 'SARIMAX-M3')\n"
    "print('완료!')",
    "fit-all"
))

# ─── 13. 잔차 진단 (CH08 패턴) ───────────────────────────────
cells.append(code(
    "for res, label in [(res_sarima,'SARIMA'), (res_m1,'M1'), (res_m2,'M2'), (res_m3,'M3')]:\n"
    "    fig = res.plot_diagnostics(figsize=(12, 7))\n"
    "    plt.suptitle(f'{label} 잔차 진단', fontsize=12, y=1.01)\n"
    "    plt.tight_layout()\n"
    "    plt.savefig(f'fig_diag_{label.lower()}.png', dpi=150)\n"
    "    plt.show()\n"
    "    lb = acorr_ljungbox(res.resid, lags=np.arange(1,11,1))\n"
    "    print(f'{label} Ljung-Box p:', np.round(lb['lb_pvalue'].values, 3))",
    "diagnostics"
))

# ─── 14. 검증구간 예측 & 성능 비교 ───────────────────────────
cells.append(md("## 7. 모델 성능 비교", "md-compare"))

cells.append(code(
    "def mape(y_true, y_pred):\n"
    "    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100\n"
    "\n"
    "steps  = len(valid)\n"
    "actual = valid['load'].values\n"
    "\n"
    "pred_sarima = res_sarima.forecast(steps=steps)\n"
    "pred_m1     = res_m1.forecast(steps=steps, exog=valid[EXOG['M1']])\n"
    "pred_m2     = res_m2.forecast(steps=steps, exog=valid[EXOG['M2']])\n"
    "pred_m3     = res_m3.forecast(steps=steps, exog=valid[EXOG['M3']])\n"
    "\n"
    "results = {}\n"
    "for label, res, pred in [\n"
    "    ('SARIMA',     res_sarima, pred_sarima),\n"
    "    ('SARIMAX-M1', res_m1,     pred_m1),\n"
    "    ('SARIMAX-M2', res_m2,     pred_m2),\n"
    "    ('SARIMAX-M3', res_m3,     pred_m3),\n"
    "]:\n"
    "    results[label] = {\n"
    "        'AIC' : res.aic,\n"
    "        'RMSE': np.sqrt(mean_squared_error(actual, pred.values)),\n"
    "        'MAPE': mape(actual, pred.values),\n"
    "        'pred': pred,\n"
    "    }\n"
    "\n"
    "print('=' * 58)\n"
    "print('모델 성능 비교 (AIC / 검증 RMSE / MAPE)')\n"
    "print('=' * 58)\n"
    "print(f'{\"모델\":<14} {\"AIC\":>10} {\"RMSE(GW)\":>10} {\"MAPE(%)\":>10}')\n"
    "print('-' * 58)\n"
    "for k, v in results.items():\n"
    "    print(f'{k:<14} {v[\"AIC\"]:>10.2f} {v[\"RMSE\"]:>10.4f} {v[\"MAPE\"]:>10.2f}')",
    "compare"
))

# ─── 15. Fig6: 검증구간 시각화 ───────────────────────────────
cells.append(code(
    "fig, ax = plt.subplots(figsize=(15, 5))\n"
    "ax.plot(valid.index, actual, 'k-', lw=1.5, label='실측')\n"
    "\n"
    "styles = {'SARIMA':'b--', 'SARIMAX-M1':'g-.', 'SARIMAX-M2':'r-', 'SARIMAX-M3':'m:'}\n"
    "for label, st in styles.items():\n"
    "    v = results[label]\n"
    "    ax.plot(valid.index, v['pred'].values, st, lw=1.2,\n"
    "            label=f'{label}  RMSE={v[\"RMSE\"]:.3f}  MAPE={v[\"MAPE\"]:.2f}%')\n"
    "\n"
    "ax.axvspan(valid.index[0], valid.index[-1], color='#808080', alpha=0.07)\n"
    "ax.set_ylabel('전력수요 (GW)')\n"
    "ax.set_title(f'검증구간 예측 비교  ({VALID_START} ~ {VALID_END})', fontsize=13)\n"
    "ax.legend(fontsize=9)\n"
    "ax.spines['top'].set_alpha(0)\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig6_validation.png', dpi=150)\n"
    "plt.show()",
    "fig6"
))

# ─── 16. Fig7: RMSE/MAPE 막대 (CH08 패턴) ────────────────────
cells.append(code(
    "labels = list(results.keys())\n"
    "rmses  = [results[k]['RMSE'] for k in labels]\n"
    "mapes  = [results[k]['MAPE'] for k in labels]\n"
    "colors = ['royalblue', 'seagreen', 'crimson', 'darkorange']\n"
    "\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))\n"
    "\n"
    "bars1 = ax1.bar(labels, rmses, color=colors, width=0.5)\n"
    "ax1.set_ylabel('RMSE (GW)')\n"
    "ax1.set_title('RMSE 비교')\n"
    "ax1.set_ylim(0, max(rmses)*1.3)\n"
    "for bar, v in zip(bars1, rmses):\n"
    "    ax1.text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v:.3f}', ha='center', fontweight='bold', fontsize=9)\n"
    "\n"
    "bars2 = ax2.bar(labels, mapes, color=colors, width=0.5)\n"
    "ax2.set_ylabel('MAPE (%)')\n"
    "ax2.set_title('MAPE 비교')\n"
    "ax2.set_ylim(0, max(mapes)*1.3)\n"
    "for bar, v in zip(bars2, mapes):\n"
    "    ax2.text(bar.get_x()+bar.get_width()/2, v+0.1, f'{v:.2f}%', ha='center', fontweight='bold', fontsize=9)\n"
    "\n"
    "plt.suptitle('4개 모델 성능 비교 (여름 검증구간)', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig7_bar_compare.png', dpi=150)\n"
    "plt.show()",
    "fig7"
))

# ─── 17. 외생변수 계수 비교 (CH09 패턴) ──────────────────────
cells.append(md("## 8. 외생변수 계수 해석", "md-coef"))

cells.append(code(
    "coef_info = {\n"
    "    'SARIMAX-M1': (res_m1,  ['CDD_avg','HDD_avg']),\n"
    "    'SARIMAX-M2': (res_m2,  ['CDD_max','HDD_min']),\n"
    "    'SARIMAX-M3': (res_m3,  ['CDD_max','HDD_min','TROP']),\n"
    "}\n"
    "\n"
    "for model_name, (res, cols) in coef_info.items():\n"
    "    print(f'\\n{'='*56}')\n"
    "    print(f'{model_name} 외생변수 계수')\n"
    "    print(f'{'='*56}')\n"
    "    print(f'{\"변수\":<12} {\"계수(GW/도일)\":>14} {\"p-value\":>10} {\"유의\":>5}')\n"
    "    print('-'*56)\n"
    "    for col in cols:\n"
    "        coef = res.params[col]\n"
    "        pval = res.pvalues[col]\n"
    "        sig  = '***' if pval<0.001 else ('**' if pval<0.01 else ('*' if pval<0.05 else 'n.s.'))\n"
    "        print(f'{col:<12} {coef:>14.4f} {pval:>10.4f} {sig:>5}')",
    "coef"
))

# ─── 18. Fig8: 계수 비교 히트맵 ──────────────────────────────
cells.append(code(
    "fig, axes = plt.subplots(1, 3, figsize=(14, 4))\n"
    "\n"
    "for ax, (model_name, (res, cols)) in zip(axes, coef_info.items()):\n"
    "    coefs = [res.params[c] for c in cols]\n"
    "    pvals = [res.pvalues[c] for c in cols]\n"
    "    bar_colors = ['crimson' if c>0 else 'steelblue' for c in coefs]\n"
    "    bars = ax.bar(cols, coefs, color=bar_colors, width=0.5)\n"
    "    ax.axhline(0, color='black', lw=0.8)\n"
    "    ax.set_title(model_name)\n"
    "    ax.set_ylabel('계수 (GW/도일)')\n"
    "    for bar, v, pv in zip(bars, coefs, pvals):\n"
    "        sig = '***' if pv<0.001 else ('**' if pv<0.01 else ('*' if pv<0.05 else ''))\n"
    "        ax.text(bar.get_x()+bar.get_width()/2,\n"
    "                v + (0.01 if v>=0 else -0.03),\n"
    "                f'{v:+.3f}{sig}', ha='center', fontsize=9, fontweight='bold')\n"
    "    ax.tick_params(axis='x', labelsize=8)\n"
    "\n"
    "plt.suptitle('모델별 외생변수 계수 비교 (*** p<0.001)', fontsize=12)\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig8_coef_compare.png', dpi=150)\n"
    "plt.show()",
    "fig8"
))

# ─── 19. 최적 모델 선택 & 전체 재학습 ───────────────────────
cells.append(md("## 9. 2026년 상반기 예측", "md-forecast"))

cells.append(code(
    "# AIC 기준 최적 모델 자동 선택\n"
    "best_name = min(results, key=lambda k: results[k]['AIC'])\n"
    "best_exog = EXOG.get(best_name.replace('SARIMAX-',''), [])\n"
    "print(f'최적 모델: {best_name}  (AIC={results[best_name][\"AIC\"]:.2f})')\n"
    "\n"
    "full = df.loc[:'2025-12-31']\n"
    "\n"
    "print('SARIMA 전체 재학습...')\n"
    "res_sarima_f = SARIMAX(full['load'],\n"
    "                       order=(p,d,q), seasonal_order=(P,D,Q,s),\n"
    "                       simple_differencing=False).fit(disp=False)\n"
    "\n"
    "print(f'{best_name} 전체 재학습...')\n"
    "res_best_f = SARIMAX(full['load'], full[best_exog],\n"
    "                     order=(p,d,q), seasonal_order=(P,D,Q,s),\n"
    "                     simple_differencing=False).fit(disp=False)\n"
    "\n"
    "steps_fc = len(exog_2026[best_name.replace('SARIMAX-','')])\n"
    "fc_sarima = res_sarima_f.get_forecast(steps=steps_fc)\n"
    "fc_best   = res_best_f.get_forecast(\n"
    "    steps=steps_fc,\n"
    "    exog=exog_2026[best_name.replace('SARIMAX-','')])\n"
    "\n"
    "idx_2026    = exog_2026[best_name.replace('SARIMAX-','')].index\n"
    "pred_s_2026 = fc_sarima.predicted_mean\n"
    "pred_b_2026 = fc_best.predicted_mean\n"
    "ci_b_2026   = fc_best.conf_int(alpha=0.05)\n"
    "pred_s_2026.index = pred_b_2026.index = ci_b_2026.index = idx_2026\n"
    "\n"
    "print(f'예측 완료: {idx_2026.min().date()} ~ {idx_2026.max().date()}')",
    "forecast"
))

# ─── 20. Fig9: 최종 예측 시각화 ──────────────────────────────
cells.append(code(
    "fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(15, 8))\n"
    "\n"
    "# 위: 2026 기온\n"
    "ax1.plot(t26.index, t26['평균기온'],  color='gray',      lw=1.0, label='평균기온')\n"
    "ax1.plot(t26.index, t26['최고기온'],  color='tomato',    lw=0.8, alpha=0.7, label='최고기온')\n"
    "ax1.plot(t26.index, t26['최저기온'],  color='steelblue', lw=0.8, alpha=0.7, label='최저기온')\n"
    "ax1.axhline(18, color='navy',      ls='--', lw=0.8, label='난방기준 18℃')\n"
    "ax1.axhline(24, color='darkorange',ls='--', lw=0.8, label='냉방기준 24℃')\n"
    "ax1.set_ylabel('기온 (℃)')\n"
    "ax1.set_title('2026 전국 기온 (SARIMAX 외생변수 입력값)')\n"
    "ax1.legend(loc='lower right', fontsize=8)\n"
    "ax1.spines['top'].set_alpha(0)\n"
    "\n"
    "# 아래: 예측\n"
    "recent = df.loc['2025-07-01':]\n"
    "ax2.plot(recent.index, recent['load'], 'k-', lw=1.2, label='실측 (2025 하반기)')\n"
    "ax2.plot(pred_s_2026.index, pred_s_2026.values,\n"
    "         'b--', lw=1.3, label='SARIMA (기온 무관, 패턴 연장)')\n"
    "ax2.plot(pred_b_2026.index, pred_b_2026.values,\n"
    "         'r-',  lw=1.3, label=f'{best_name} (기온 굴곡 추종)')\n"
    "ax2.fill_between(ci_b_2026.index,\n"
    "                 ci_b_2026.iloc[:,0], ci_b_2026.iloc[:,1],\n"
    "                 color='crimson', alpha=0.15, label='95% CI')\n"
    "ax2.axvline(x=df.index[-1], color='gray', ls=':', lw=1.5, label='예측 시작')\n"
    "ax2.set_ylabel('전력수요 (GW)')\n"
    "ax2.set_title(f'2026 상반기 전력수요 예측: SARIMA vs {best_name}', fontsize=13)\n"
    "ax2.legend(fontsize=9)\n"
    "ax2.spines['top'].set_alpha(0)\n"
    "\n"
    "fig.autofmt_xdate()\n"
    "plt.tight_layout()\n"
    "plt.savefig('fig9_forecast_2026.png', dpi=150)\n"
    "plt.show()",
    "fig9"
))

# ─── 21. 결론 ─────────────────────────────────────────────────
cells.append(md(
    "## 10. 결론\n"
    "\n"
    "### 모델 비교 요약\n"
    "\n"
    "| 모델 | 외생변수 | AIC | RMSE | MAPE |\n"
    "|------|---------|-----|------|------|\n"
    "| SARIMA | — | (결과) | (결과) | (결과) |\n"
    "| SARIMAX-M1 | CDD_avg, HDD_avg | (결과) | (결과) | (결과) |\n"
    "| SARIMAX-M2 | CDD_max, HDD_min | (결과) | (결과) | (결과) |\n"
    "| SARIMAX-M3 | + TROP | (결과) | (결과) | (결과) |\n"
    "\n"
    "### 핵심 해석\n"
    "- **CDD_max > CDD_avg** : 낮 더위(최고기온)가 냉방 수요를 더 잘 설명\n"
    "- **HDD_min > HDD_avg** : 새벽 추위(최저기온)가 난방 수요를 더 잘 설명\n"
    "- **TROP** : 열대야 효과 — AIC 감소 여부로 채택 결정\n"
    "- SARIMAX는 2026년 기온 굴곡(1월 한파, 5월 온화)을 예측에 직접 반영",
    "md-conclusion"
))

# ── JSON 저장 ────────────────────────────────────────────────
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"codemirror_mode": {"name": "ipython", "version": 3},
                          "file_extension": ".py", "mimetype": "text/x-python",
                          "name": "python", "version": "3.12.4"}
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analysis_elec.ipynb')
with open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f'생성 완료: {out}')
