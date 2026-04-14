from sklearn.ensemble import RandomForestRegressor


def train_ai_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def predict_ai_model(model, X):
    return model.predict(X)