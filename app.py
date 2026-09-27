"""
app.py
Flask backend for the AI-Based Student Performance Prediction & Early Warning System.
"""

import os
import json
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, render_template, request, jsonify, send_from_directory

APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")
HISTORY_PATH = os.path.join(APP_DIR, "student_history.json")

app = Flask(__name__)

model_bundle = None
if os.path.exists(MODEL_PATH):
    model_bundle = joblib.load(MODEL_PATH)
else:
    print("WARNING: model.pkl not found. Run `python train_model.py` first.")

AT_RISK_LABEL = "At-Risk"


def load_history():
    if not os.path.exists(HISTORY_PATH):
        return []
    with open(HISTORY_PATH, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return []


def save_history(records):
    with open(HISTORY_PATH, "w") as f:
        json.dump(records, f, indent=2)


def make_prediction(attendance, previous_grade, study_hours, assignments_completed):
    if model_bundle is None:
        raise RuntimeError("Model not loaded. Run train_model.py first.")

    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    features = model_bundle["features"]

    row = pd.DataFrame(
        [[attendance, previous_grade, study_hours, assignments_completed]],
        columns=features,
    )

    pred_encoded = model.predict(row)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(row)[0]
    confidence = round(float(max(proba)) * 100, 2)

    is_at_risk = pred_label == AT_RISK_LABEL

    return {
        "performance": pred_label,
        "confidence": confidence,
        "at_risk": is_at_risk,
    }

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(silent=True) or request.form

        name = data.get("name", "Unnamed Student")
        attendance = float(data.get("attendance"))
        previous_grade = float(data.get("previous_grade"))
        study_hours = float(data.get("study_hours"))
        assignments_completed = float(data.get("assignments_completed"))

        result = make_prediction(
            attendance, previous_grade, study_hours, assignments_completed
        )

        record = {
            "name": name,
            "attendance": attendance,
            "previous_grade": previous_grade,
            "study_hours": study_hours,
            "assignments_completed": assignments_completed,
            "performance": result["performance"],
            "confidence": result["confidence"],
            "at_risk": result["at_risk"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        history = load_history()
        history.insert(0, record)
        save_history(history)

        if request.is_json:
            return jsonify(record)
        return render_template("result.html", record=record)

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 400
        return render_template("result.html", error=str(e))


@app.route("/history")
def history():
    records = load_history()
    return render_template("history.html", records=records)


@app.route("/api/history")
def api_history():
    return jsonify(load_history())


@app.route("/manifest.json")
def manifest():
    return send_from_directory(os.path.join(APP_DIR, "static"), "manifest.json")


@app.route("/service-worker.js")
def service_worker():
    return send_from_directory(os.path.join(APP_DIR, "static"), "service-worker.js")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
