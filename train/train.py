import os

from src.data_loader import load_dataset
from src.preprocessing import (
    prepare_features_and_labels,
    encode_labels,
    split_data,
    scale_features
)
from src.model import get_candidate_models, train_and_select_best_model
from src.evaluation import print_classification_results, save_confusion_matrix
from src.utils import save_artifacts


def main():
    data_path = "data/landmarks_features/all.csv"

    model_dir = "models"
    docs_dir = "docs"

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "posture_classifier.pkl")
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    encoder_path = os.path.join(model_dir, "label_encoder.pkl")
    confusion_matrix_path = os.path.join(docs_dir, "confusion_matrix.png")

    # Load dataset
    df = load_dataset(data_path)

    # Prepare data
    X, y = prepare_features_and_labels(df)
    y_encoded, label_encoder = encode_labels(y)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y_encoded)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Train models
    models = get_candidate_models()
    best_model_name, best_model, best_predictions, best_accuracy = train_and_select_best_model(
        models,
        X_train_scaled,
        y_train,
        X_test_scaled,
        y_test
    )

    # Print report
    print_classification_results(y_test, best_predictions, label_encoder)

    # Save confusion matrix
    save_confusion_matrix(
        y_test,
        best_predictions,
        label_encoder,
        confusion_matrix_path
    )

    # Save artifacts
    save_artifacts(
        best_model,
        scaler,
        label_encoder,
        model_path,
        scaler_path,
        encoder_path
    )

    # Success check
    if best_accuracy >= 0.80:
        print("\nSUCCESS: Minimum acceptable accuracy (>= 80%) achieved.")
    else:
        print("\nWARNING: Accuracy is below 80%.")


if __name__ == "__main__":
    main()
