import joblib


def save_artifacts(model, scaler, label_encoder, model_path, scaler_path, encoder_path):
    """
    Save model artifacts.
    """
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(label_encoder, encoder_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print(f"Label encoder saved to: {encoder_path}")
