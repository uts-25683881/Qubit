from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def prepare_features_and_labels(df):
    """
    Drop non-feature columns and return X, y.
    """
    if "class" not in df.columns:
        raise ValueError("Expected target column 'class' not found.")

    X = df.drop(columns=["filename", "class"], errors="ignore")
    y = df["class"]

    return X, y


def encode_labels(y):
    """
    Encode string labels into numeric labels.
    """
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print("\nClass mapping:")
    for idx, label in enumerate(label_encoder.classes_):
        print(f"{label} -> {idx}")

    return y_encoded, label_encoder


def split_data(X, y, test_size=0.30, random_state=42):
    """
    Perform 70/30 stratified train-test split.
    """
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
    """
    Fit scaler on train only, transform both train and test.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
