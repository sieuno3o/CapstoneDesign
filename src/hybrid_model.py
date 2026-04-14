from sklearn.ensemble import GradientBoostingRegressor


def train_hybrid_model(X_train, y_train):
    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


def predict_hybrid_model(model, X):
    return model.predict(X)