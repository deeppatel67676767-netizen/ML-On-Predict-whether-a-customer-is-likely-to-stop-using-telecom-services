"""
Train the four ML models on preprocessed data.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models():
    """Return a dict of model name -> sklearn estimator."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=42)
    }


def train_all_models(X_train, y_train) -> dict:
    """Fit all models on training data. Returns dict of name -> fitted model."""
    models = get_models()
    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model
    return trained
