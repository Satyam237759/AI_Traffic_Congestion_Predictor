"""Create a reproducible sample dataset and train a traffic classifier.

The generated data is synthetic for learning/demo purposes. It is NOT live
traffic data and predictions should not be used for real route decisions.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

RANDOM_STATE = 42

def make_dataset(n=2500):
    rng = np.random.default_rng(RANDOM_STATE)
    hours = rng.integers(0, 24, n)
    days = rng.choice(["Monday", "Tuesday", "Wednesday", "Thursday",
                       "Friday", "Saturday", "Sunday"], n)
    vehicles = rng.integers(50, 1201, n)
    weather = rng.choice(["Clear", "Cloudy", "Rainy", "Foggy"], n,
                         p=[0.48, 0.25, 0.20, 0.07])
    road = rng.choice(["Highway", "Main Road", "Residential"], n,
                      p=[0.25, 0.50, 0.25])

    # Rush-hour, weekday and weather effects are simulated for the demo.
    rush = np.where(((hours >= 7) & (hours <= 10)) |
                    ((hours >= 17) & (hours <= 20)), 190, 0)
    weekend = np.where(np.isin(days, ["Saturday", "Sunday"]), -65, 0)
    weather_effect = pd.Series(weather).map(
        {"Clear": 0, "Cloudy": 25, "Rainy": 115, "Foggy": 85}).to_numpy()
    road_effect = pd.Series(road).map(
        {"Highway": -35, "Main Road": 45, "Residential": -15}).to_numpy()
    score = vehicles * 0.62 + rush + weekend + weather_effect + road_effect
    score += rng.normal(0, 90, n)
    labels = np.where(score < 470, "Low",
                      np.where(score < 850, "Medium", "High"))

    return pd.DataFrame({
        "hour": hours, "day": days, "vehicles": vehicles,
        "weather": weather, "road_type": road, "congestion": labels
    })

def train():
    data = make_dataset()
    X = data[["hour", "day", "vehicles", "weather", "road_type"]]
    y = data["congestion"]
    categorical = ["day", "weather", "road_type"]
    numeric = ["hour", "vehicles"]
    preprocessor = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", "passthrough", numeric)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=180, random_state=RANDOM_STATE,
            class_weight="balanced", min_samples_leaf=2
        ))
    ])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    print("Validation accuracy on synthetic test data:",
          round(accuracy_score(y_test, predictions) * 100, 2), "%")
    print(classification_report(y_test, predictions))
    joblib.dump(pipeline, "traffic_model.pkl")
    data.to_csv("traffic_sample.csv", index=False)
    print("Saved traffic_model.pkl and traffic_sample.csv")

if __name__ == "__main__":
    train()
