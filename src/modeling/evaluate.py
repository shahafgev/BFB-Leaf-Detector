from sklearn.metrics import classification_report

def evaluate_model(model, X_test, y_test, name=""):
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    return model.score(X_test, y_test)

