from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def get_model_grid():
    return {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=2000),
            "params": {
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "penalty": ['l1', 'l2'],
                "solver": ['liblinear', 'saga']
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [50, 100],
                "criterion": ["gini", "entropy"],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
                "min_samples_leaf": [2, 5, 10]
            }
        }
        # ,
        # "SVM": {
        #     "model": SVC(),
        #     "params": {
        #         "C": [0.1, 1, 10],
        #         "kernel": ["linear", "rbf"],
        #         "gamma": ['scale', 0.1]
        #     }
        # }
    }

