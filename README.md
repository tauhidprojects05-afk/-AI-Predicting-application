# Student Performance AI

A simple web app that predicts how a student is likely to perform (Excellent, Good, Average, or At-Risk) based on their attendance, previous grades, study hours, and assignment completion. It also flags at-risk students early so a teacher can step in.

Built with **Flask** (Python) for the backend and a trained **machine learning model** (scikit-learn) for the predictions.

---

## What it does

1. You enter a student's details in a form (name, attendance %, previous grade %, weekly study hours, assignments completed %).
2. The app runs those numbers through a trained AI model.
3. It shows a result: the predicted performance level, a confidence score, and a warning if the student is at risk.
4. Every prediction is saved, so you can look back at a full history of past predictions.

---

## Project files

```
your-project/
├── app.py                 → the Flask web server (main app)
├── train_model.py         → trains the AI model and saves it as model.pkl
├── model.pkl              → the trained model (created by train_model.py)
├── student_data.csv       → the data used to train the model
├── requirements.txt       → list of Python packages this project needs
├── student_history.json   → auto-created file that stores past predictions
├── static/
│   ├── style.css           → all the visual styling
│   ├── manifest.json       → makes the app installable like a mobile app
│   └── service-worker.js   → lets the app work offline / cache files
└── templates/
    ├── base.html            → shared page layout (navbar, footer)
    ├── index.html           → the prediction form page
    ├── result.html          → the page that shows a prediction result
    └── history.html         → the page that lists all past predictions
```

---

## How to run it on your computer

### 1. Make sure Python is installed
You need Python 3.9 or newer. Check by running:
```
python --version
```

### 2. Install the required packages
Open a terminal in the project folder and run:
```
pip install -r requirements.txt
```

### 3. Train the model (only needed once, or when student_data.csv changes)
```
python train_model.py
```
This reads `student_data.csv`, trains the AI model, and saves it as `model.pkl`.

### 4. Start the app
```
python app.py
```

### 5. Open it in your browser
Go to:
```
http://127.0.0.1:5000
```
or
```
http://localhost:5000
```

---

## How the prediction works

- The AI model looks at four things: **attendance**, **previous grade**, **weekly study hours**, and **assignments completed**.
- It's a **Random Forest** model — a common, reliable type of machine learning model for this kind of task.
- It was trained on the data in `student_data.csv`, learning patterns from real examples of past student outcomes.
- If a student is predicted as "At-Risk," the app shows an early-warning message recommending intervention.

---

## Where past predictions are stored

Every time someone makes a prediction, it's saved to `student_history.json`. You can view all of them on the **History** page in the app, or fetch them as raw data at:
```
http://127.0.0.1:5000/api/history
```

---

## Notes

- `debug=True` in `app.py` means the app automatically reloads when you edit code, and shows detailed errors — turn this off before putting the app on a real server.
- If you change `static/style.css` or `static/service-worker.js` and don't see your changes in the browser, go to your browser's DevTools → Application tab → Service Workers → click "Unregister," then refresh the page. This clears old cached files.
- If port 5000 is already being used by something else on your computer, you'll get an "address already in use" error. You can fix this by changing `port=5000` to a different number (like `5001`) in `app.py`.

---

## Retraining with new data

If you want the model to learn from new or updated student records:

1. Update `student_data.csv` with new rows.
2. Run `python train_model.py` again. This creates a fresh `model.pkl`.
3. Restart the app (`python app.py`) so it loads the newly trained model.
