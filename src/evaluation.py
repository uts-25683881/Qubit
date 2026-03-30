import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


def print_classification_results(y_true, y_pred, label_encoder):
    """
    Print classification report.
    """
    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=label_encoder.classes_
        )
    )


def save_confusion_matrix(y_true, y_pred, label_encoder, output_path):
    """
    Save confusion matrix plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=label_encoder.classes_
    )

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")
