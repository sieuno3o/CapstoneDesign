import pandas as pd
import pmdarima as pm
from pathlib import Path
import matplotlib.pyplot as plt

def find_best_arima_model(series: pd.Series, seasonal: bool = False, m: int = 1):
    """
    pmdarima의 auto_arima를 이용하여 주어진 시계열 데이터에 대한 최적의 ARIMA(p,d,q) 파라미터를 탐색합니다.
    - seasonal: 계절성 모델(SARIMA) 사용 여부 (기본값 False)
    - m: 계절성 주기 (seasonal=True일 때 사용, 예: 분기=4, 월=12)
    """
    print(f"\n--- [Auto ARIMA] 파라미터 탐색 시작 ---")
    print(f"데이터 크기: {len(series)}")
    
    # auto_arima를 통해 가장 낮은 AIC를 가지는 모델을 자동으로 찾음
    # (내부적으로 KPSS 및 ADF 테스트를 통해 최적의 d 값을 추정함)
    model = pm.auto_arima(
        series,
        start_p=0, max_p=5,
        d=None,     # 자동으로 d 값을 찾음
        start_q=0, max_q=5,
        seasonal=seasonal,
        m=m,
        test='adf', # d 값을 찾기 위해 ADF 테스트 사용
        trace=True, # 탐색 과정 출력
        error_action='ignore',  
        suppress_warnings=True, 
        stepwise=True # 시간 단축을 위해 단계별 탐색
    )
    
    print("\n--- [Auto ARIMA] 최적 모델 탐색 완료 ---")
    print(model.summary())
    
    return model

def plot_acf_pacf_diagnostics(series: pd.Series, save_dir="results/figures"):
    """
    자기상관함수(ACF) 및 편자기상관함수(PACF) 시각화를 통한 파라미터 후보군 수동 분석용 함수
    """
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 4))
    
    plot_acf(series.dropna(), ax=axes[0], lags=40, title="Autocorrelation (ACF) - q 파라미터 추정")
    plot_pacf(series.dropna(), ax=axes[1], lags=40, title="Partial Autocorrelation (PACF) - p 파라미터 추정")
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "acf_pacf_plot.png"
    plt.savefig(save_path)
    print(f"[INFO] ACF/PACF 플롯을 저장했습니다: {save_path}")
    plt.close()
