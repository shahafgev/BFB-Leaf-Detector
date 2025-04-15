import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model, X_test, y_test, name=""):
    print(f"\n--- {name} ---")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    # Create and save confusion matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Create outputs directory if it doesn't exist
    os.makedirs('outputs', exist_ok=True)
    
    # Save the plot
    plt.savefig(f'outputs/confusion_matrix_{name.lower().replace(" ", "_")}.png')
    plt.close()
    
    return model.score(X_test, y_test)

