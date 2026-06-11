from sklearn.linear_model import LinearRegression


def train_sentiment_only_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def predict_sentiment_only_model(model, X):
    return model.predict(X)