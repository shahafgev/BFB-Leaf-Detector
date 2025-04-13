import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from src.modeling.train import get_model_grid
from src.modeling.evaluate import evaluate_model

# Load data
df = pd.read_csv("processed_data/pixel_dataset_rgbhsv.csv")
X = df[["B", "G", "R", "H", "S", "V"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_grid = get_model_grid()

best_model = None
best_score = 0
best_name = ""

for name, item in model_grid.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(item["model"], item["params"], cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    print(f"Best parameters for {name}: {grid.best_params_}")
    score = evaluate_model(grid.best_estimator_, X_test, y_test, name)

    if score > best_score:
        best_model = grid.best_estimator_
        best_score = score
        best_name = name

joblib.dump(best_model, "models/best_model.pkl")
print(f"\n✅ Best model: {best_name} (accuracy: {best_score:.4f}) saved to models/best_model.pkl")
