import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf
from pathlib import Path

def evaluate_residuals(model, save_dir="results/figures"):
    """
    ARIMA 모델의 잔차(Residual)가 백색 잡음(White Noise) 형태를 띠는지 확인하는 함수.
    1. Ljung-Box Test (잔차 간 자기상관성이 있는지 검정)
    2. 잔차 시계열 플롯
    3. 잔차의 분포 (히스토그램)
    4. 잔차의 자기상관성 (ACF)
    """
    # 🌟 모델 프레임워크(pmdarima 혹은 statsmodels)에 호환될 수 있게 잔차 추출 🌟
    if hasattr(model, "resid"):
        residuals = pd.Series(model.resid())
    else:
        # ARIMA 등 다른 statsmodels 객체인 경우
        residuals = pd.Series(model.resid)
        
    print("\n--- [Diagnostics] 잔차 진단 시작 ---")
    
    # 1. Ljung-Box Test
    # 귀무가설(H0): 잔차들이 백색 잡음이다. (자기 상관성이 없다)
    # 대립가설(H1): 잔차들 간 자기 상관성이 존재한다.
    # p-value > 0.05 이면 좋은 모델.
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    p_value = lb_test['lb_pvalue'].iloc[0]
    
    print("=== Ljung-Box Test Results ===")
    print(lb_test)
    if p_value > 0.05:
        print(f"[성공] p-value({p_value:.4f}) > 0.05. 잔차 간 자기상관이 없습니다 (백색잡음 가정 충족).")
    else:
        print(f"[주의] p-value({p_value:.4f}) <= 0.05. 잔차에 패턴이 남아있습니다. 모델 파라미터 개선이 필요할 수 있습니다.")

    # 2. 잔차 시각화 플롯 저장
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # 잔차의 시계열 플롯
    axes[0].plot(residuals)
    axes[0].set_title('Residuals Over Time (백색 잡음과 같은지 확인)')
    axes[0].axhline(0, color='r', linestyle='--')
    
    # 잔차의 히스토그램 (정규분포를 띄어야 좋음)
    axes[1].hist(residuals, bins=40, density=True, alpha=0.6, color='b')
    axes[1].set_title('Density of Residuals (정규분포를 따르는지 확인)')
    
    # 잔차의 ACF 플롯 (시차가 커질 때 파란 점선 안에 들어와야 함)
    plot_acf(residuals, ax=axes[2], lags=40, title='ACF of Residuals (패턴이 남아았는지 확인)')
    
    # 저장
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(save_dir) / "residual_diagnostics.png"
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[INFO] 잔차 진단 플롯 시각화를 저장했습니다: {save_path}")
    plt.close()
