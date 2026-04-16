import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, mean_absolute_percentage_error


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    print("\n=== Model Prediction Metrics ===")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print(f"  MAPE : {mape:.4f}%")
    
    return {"mae": mae, "rmse": rmse, "mape": mape}


def direction_accuracy(y_true, y_pred):
    """
    등락(방향성) 맞춤 여부 계산 (현재값 대비 다음 날 상승/하락 여부)
    """
    # 실제값 방향성 (t 대비 t+1)
    true_diff = np.diff(y_true)
    true_dir = (true_diff > 0).astype(int)
    
    # 예측값 방향성 (t 대비 t+1)
    pred_diff = np.diff(y_pred)
    pred_dir = (pred_diff > 0).astype(int)
    
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