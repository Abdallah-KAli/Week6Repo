import os
import pickle
import pandas as pd

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns

from src.preprocessing import clean_text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data", "data.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "models", "confusion_matrix.png")


def evaluate():
    df = pd.read_csv(DATA_PATH).dropna()

    df["text"] = df["text"].apply(clean_text)

    X = df["text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    plt.savefig(OUTPUT_PATH)

    print("Confusion matrix saved at:", OUTPUT_PATH)