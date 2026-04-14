import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


def regression_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
    }


def direction_accuracy(y_true_direction, y_pred_direction):
    return accuracy_score(y_true_direction, y_pred_direction)