"""
train_model.py
Trains a RandomForest classifier to predict student performance category
from attendance, previous grades, study hours, and assignment completion.

Run this once (or whenever student_data.csv is updated) to regenerate model.pkl:
    python train_model.py
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib

FEATURES = ["attendance", "previous_grade", "study_hours", "assignments_completed"]
TARGET = "performance"
DATA_PATH = "student_data.csv"
MODEL_PATH = "model.pkl"


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\nTest Accuracy: {acc:.2%}\n")
    print(classification_report(y_test, preds, target_names=label_encoder.classes_))

    bundle = {
        "model": model,
        "label_encoder": label_encoder,
        "features": FEATURES,
    }
    joblib.dump(bundle, MODEL_PATH)
    print(f"Saved trained model bundle to '{MODEL_PATH}'")


if __name__ == "__main__":
    main()
