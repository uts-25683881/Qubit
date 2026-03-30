from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def get_candidate_models():
    """
    Return candidate models.
    """
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", random_state=42),
    }


def train_and_select_best_model(models, X_train, y_train, X_test, y_test):
    """
    Train all models and select the best one using test accuracy.
    """
    best_model_name = None
    best_model = None
    best_predictions = None
    best_accuracy = 0.0

    print("\nTraining models...\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"{name} Accuracy: {acc:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            best_model_name = name
            best_model = model
            best_predictions = y_pred

    print(f"\nBest model: {best_model_name}")
    print(f"Best accuracy: {best_accuracy:.4f}")

    return best_model_name, best_model, best_predictions, best_accuracy
