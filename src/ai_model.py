import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping


def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_ann_model(X_train, y_train, X_val=None, y_val=None):
    """
    ANN 모델 학습.
    - Dropout(0.2): 과적합 방지
    - EarlyStopping(patience=15): 검증 손실이 개선되지 않으면 조기 종료
    - X_val, y_val: 외부 검증 데이터 (val_df) 사용 시 전달
    """
    model = Sequential([
        Input(shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=0
    )

    if X_val is not None and y_val is not None:
        # 외부 val_df를 validation_data로 사용 (권장)
        validation_data = (X_val, y_val)
        validation_split = 0.0
    else:
        # val_df가 없을 경우 train 내부에서 10% 분리
        validation_data = None
        validation_split = 0.1

    model.fit(
        X_train, y_train,
        epochs=200,
        batch_size=16,
        validation_data=validation_data,
        validation_split=validation_split,
        callbacks=[early_stop],
        verbose=0
    )
    return model


def predict_ai_model(model, X):
    pred = model.predict(X, verbose=0)
    # Keras returns 2D array, RF returns 1D array
    if len(pred.shape) > 1 and pred.shape[1] == 1:
        pred = pred.flatten()
    return pred
