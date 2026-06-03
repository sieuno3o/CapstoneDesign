import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, mean_absolute_percentage_error, r2_score


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    mbe = np.mean(y_pred - y_true)
    r2 = r2_score(y_true, y_pred)
    
    print("\n=== Model Prediction Metrics ===")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.4f}%")
    print(f"  MBE  : {mbe:.4f}")
    print(f"  R²   : {r2:.4f}")
    
    return {"rmse": rmse, "mae": mae, "mape": mape, "mbe": mbe, "r2": r2}


def direction_accuracy(y_true, y_pred):
    """
    등락(방향성) 맞춤 여부 계산.
    기준: 전일 실제 종가(y_true[t-1]) 대비 내일 실제/예측 종가의 상승 여부 비교
    - 실제 방향: y_true[t] > y_true[t-1]
    - 예측 방향: y_pred[t] > y_true[t-1]  ← 전일 실제값을 기준으로 사용
    """
    # 전일 실제 종가 기준으로 비교 (t-1은 실제값, t는 예측/실제값)
    true_dir = (y_true[1:] > y_true[:-1]).astype(int)   # 실제 방향
    pred_dir = (y_pred[1:] > y_true[:-1]).astype(int)   # 예측 방향 (기준: 전일 실제값)

    acc = accuracy_score(true_dir, pred_dir)
    print(f"  방향성 정확도 (Directional Accuracy): {acc:.4f}")
    return acc

def evaluate_and_plot(y_true: pd.Series, y_pred: pd.Series, title="ARIMA Forecast vs Actual", save_dir="results/figures", data_name=""):
    """
    평가 지표를 출력하고, 실제 값과 예측 값을 비교하는 시계열 그래프를 저장합니다.
    """
    metrics = regression_metrics(y_true.values, y_pred.values)
    dir_acc = direction_accuracy(y_true.values, y_pred.values)
    
    metrics["direction_accuracy"] = dir_acc
    
    plt.figure(figsize=(14, 6))
    plt.plot(y_true.index, y_true.values, label='Actual Data', color='royalblue')
    plt.plot(y_pred.index, y_pred.values, label='Predictions', color='darkorange', linestyle='--')
    
    plt.title(title)
    plt.xlabel('Date / Index')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_name = f"forecast_vs_actual_{data_name}.png" if data_name else "forecast_vs_actual.png"
    save_path = Path(save_dir) / save_name
    plt.savefig(save_path)
    print(f"[INFO] 예측 시각화 이미지를 저장했습니다: {save_path}")
    plt.close()
    
    return metrics