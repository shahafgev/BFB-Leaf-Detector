import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from src.modeling.train import get_model_grid
from src.modeling.evaluate import evaluate_model

# Load data
print("Loading dataset...")
df = pd.read_csv("processed_data/pixel_dataset_rgbhsv.csv")
X = df[["B", "G", "R", "H", "S", "V"]]
y = df["label"]

# Calculate class weights for balanced training
class_weights = compute_class_weight('balanced', classes=df['label'].unique(), y=df['label'])
class_weight_dict = dict(zip(df['label'].unique(), class_weights))

# Split data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Get model grid
model_grid = get_model_grid()

# Update model parameters to use class weights
for name, item in model_grid.items():
    if hasattr(item["model"], "class_weight"):
        item["params"]["class_weight"] = ["balanced", class_weight_dict]

best_model = None
best_score = 0
best_name = ""

for name, item in model_grid.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(item["model"], item["params"], cv=5, scoring="accuracy", n_jobs=1)
    grid.fit(X_train, y_train)

    print(f"Best parameters for {name}: {grid.best_params_}")
    score = evaluate_model(grid.best_estimator_, X_test, y_test, name)

    if score > best_score:
        best_model = grid.best_estimator_
        best_score = score
        best_name = name

joblib.dump(best_model, "models/best_model.pkl")
print(f"\n✅ Best model: {best_name} (accuracy: {best_score:.4f}) saved to models/best_model.pkl")
