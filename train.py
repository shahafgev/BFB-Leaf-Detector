import pandas as pd
import joblib
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.utils.class_weight import compute_class_weight
from src.modeling.train import get_model_grid
from src.modeling.evaluate import evaluate_model
from sklearn.utils import resample

def balance_dataset(X, y):
    """
    Balance the dataset by downsampling the majority class.
    Returns balanced X and y.
    """
    # Combine features and target
    df = pd.concat([X, y], axis=1)
    
    # Get the class with minimum samples
    min_class = y.value_counts().idxmin()
    min_samples = y.value_counts().min()
    
    # Get majority class
    majority_class = y.value_counts().idxmax()
    
    # Separate majority and minority classes
    df_majority = df[df['label'] == majority_class]
    df_minority = df[df['label'] == min_class]
    
    # Downsample majority class
    df_majority_downsampled = resample(df_majority,
                                     replace=False,  # sample without replacement
                                     n_samples=min_samples,  # match minority class size
                                     random_state=42)  # for reproducibility
    
    # Combine minority class with downsampled majority class
    df_balanced = pd.concat([df_majority_downsampled, df_minority])
    
    # Shuffle the dataset
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Separate features and target
    X_balanced = df_balanced[["B", "G", "R", "H", "S", "V"]]
    y_balanced = df_balanced["label"]
    
    return X_balanced, y_balanced

# Load data
print("Loading dataset...")
df = pd.read_csv("processed_data/pixel_dataset_rgbhsv.csv")
X = df[["B", "G", "R", "H", "S", "V"]]
y = df["label"]

# Split data first to keep test set with original distribution
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Balance the training dataset
print("\nBalancing training dataset...")
X_train_balanced, y_train_balanced = balance_dataset(X_train, y_train)

print("\nOriginal class distribution:")
print(y_train.value_counts())
print("\nBalanced class distribution:")
print(y_train_balanced.value_counts())

# Get model grid
model_grid = get_model_grid()

# Update model parameters to use class weights (optional now since dataset is balanced)
for name, item in model_grid.items():
    if hasattr(item["model"], "class_weight"):
        item["params"]["class_weight"] = ["balanced"]  # Using 'balanced' instead of computed weights

best_model = None
best_score = 0
best_name = ""

for name, item in model_grid.items():
    print(f"\n🔍 Tuning {name}...")
    grid = GridSearchCV(item["model"], item["params"], cv=5, scoring="accuracy", n_jobs=1)
    grid.fit(X_train_balanced, y_train_balanced)  # Using balanced training data

    print(f"Best parameters for {name}: {grid.best_params_}")
    score = evaluate_model(grid.best_estimator_, X_test, y_test, name)  # Evaluating on original test set

    if score > best_score:
        best_model = grid.best_estimator_
        best_score = score
        best_name = name

joblib.dump(best_model, "models/best_model.pkl")
print(f"\n✅ Best model: {best_name} (accuracy: {best_score:.4f}) saved to models/best_model.pkl")
