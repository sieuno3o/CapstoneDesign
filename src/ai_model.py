import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def train_rf_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_ann_model(X_train, y_train):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(32, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=0)
    return model


def predict_ai_model(model, X):
    pred = model.predict(X)
    # Keras returns 2D array, RF returns 1D array
    if len(pred.shape) > 1 and pred.shape[1] == 1:
        pred = pred.flatten()
    return pred