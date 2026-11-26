from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "traffic_model.pkl")

def load_model():
    if not os.path.exists(MODEL_PATH):
        from train_model import train
        train()
    return joblib.load(MODEL_PATH)

model = load_model()

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    inputs = {}
    if request.method == "POST":
        try:
            inputs = {
                "hour": int(request.form["hour"]),
                "day": request.form["day"],
                "vehicles": int(request.form["vehicles"]),
                "weather": request.form["weather"],
                "road_type": request.form["road_type"],
            }
            row = pd.DataFrame([inputs])
            prediction = model.predict(row)[0]
            probabilities = model.predict_proba(row)[0]
            confidence = round(float(max(probabilities)) * 100, 1)
        except (ValueError, KeyError):
            prediction = "Please enter valid values."
    return render_template("index.html", prediction=prediction,
                           confidence=confidence, inputs=inputs)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == "__main__":
    app.run(debug=True)
