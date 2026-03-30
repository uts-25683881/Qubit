import argparse
import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score
)

LANDMARK_START = 0
NUM_LANDMARKS = 33

ROBOFLOW_FINE_LABELS = frozenset(
    {"leaning_backward", "leaning_forward", "leaning_left", "leaning_right", "upright"}
)


def collapse_roboflow_fine_to_binary(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    s = out["class"].astype(str)
    mask = s.isin(ROBOFLOW_FINE_LABELS)
    if not mask.any():
        return out
    upright = s == "upright"
    out.loc[mask & upright, "class"] = "correct"
    out.loc[mask & ~upright, "class"] = "incorrect"
    print("Roboflow 5-class -> binary (fine-labeled rows only)")
    print(out.loc[mask, "class"].value_counts())
    return out


def load_dataset(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset not found: {file_path}")
    df = pd.read_csv(file_path)
    print(f"Loaded dataset: {file_path}")
    print(f"Shape: {df.shape}")
    return df


def get_landmark_columns():
    cols = []
    for i in range(LANDMARK_START, NUM_LANDMARKS):
        cols.extend([f"x{i}", f"y{i}", f"z{i}", f"v{i}"])
    return cols


def prepare_features_and_labels(df):
    if "class" not in df.columns:
        raise ValueError("Expected 'class' column in dataset.")

    landmark_cols = get_landmark_columns()

    missing = [c for c in landmark_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing landmark columns: {missing[:10]}")

    X = df[landmark_cols].copy()
    y = df["class"].copy()
    return X, y


def encode_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClass mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {idx}")

    return y_encoded, label_encoder


def split_data(X, y, test_size=0.30, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    print(f"\nTrain shape: {X_train.shape}")
    print(f"Test shape: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler


def get_candidate_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=42)
    }


def train_and_select_best_model(models, X_train, y_train, X_test, y_test):
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


def print_classification_results(y_true, y_pred, label_encoder):
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))


def save_confusion_matrix(y_true, y_pred, label_encoder, output_path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Confusion matrix saved to: {output_path}")


def save_artifacts(model, scaler, label_encoder, model_path, scaler_path, encoder_path):
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)
    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Label encoder saved to: {encoder_path}")


def main(*, use_binary: bool):
    data_path = "data/landmarks/all.csv"

    model_dir = "models"
    docs_dir = "docs"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "posture_classifier.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    confusion_matrix_path = os.path.join(docs_dir, "confusion_matrix.png")

    df = load_dataset(data_path)
    if use_binary:
        df = collapse_roboflow_fine_to_binary(df)
    X, y = prepare_features_and_labels(df)
    y_encoded, label_encoder = encode_labels(y)

    X_train, X_test, y_train, y_test = split_data(X, y_encoded)
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    models = get_candidate_models()
    best_model_name, best_model, best_predictions, best_accuracy = train_and_select_best_model(
        models, X_train_scaled, y_train, X_test_scaled, y_test
    )

    print_classification_results(y_test, best_predictions, label_encoder)
    save_confusion_matrix(y_test, best_predictions, label_encoder, confusion_matrix_path)
    save_artifacts(best_model, scaler, label_encoder, model_path, scaler_path, encoder_path)

    if best_accuracy >= 0.80:
        print("\nSUCCESS: Minimum acceptable accuracy (>= 80%) achieved.")
    else:
        print("\nWARNING: Accuracy is below 80%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train posture classifier on landmarks CSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--binary",
        dest="use_binary",
        action="store_const",
        const=True,
        help="Collapse Roboflow 5 labels to correct/incorrect",
    )
    group.add_argument(
        "--multiclass",
        dest="use_binary",
        action="store_const",
        const=False,
        help="Train on raw 'class' column (e.g. upright / leaning_*)",
    )
    args = parser.parse_args()
    main(use_binary=args.use_binary)
