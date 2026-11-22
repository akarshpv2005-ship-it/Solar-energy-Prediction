
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

app = Flask(__name__)

# Load trained XGBoost model
model = joblib.load("models/xgboost_model.pkl")
features = joblib.load("models/features.pkl")

# Load original dataset to obtain valid feature ranges
df = pd.read_csv("dataset/Solar Power Plant Data.csv")

# Dataset-based defaults and ranges
wind_min = float(df["WindSpeed"].min())
wind_max = float(df["WindSpeed"].max())

temperature_min = float(df["AirTemperature"].min())
temperature_max = float(df["AirTemperature"].max())

wind_default = float(df["WindSpeed"].median())
temperature_default = float(df["AirTemperature"].median())

# Use median values for inputs not exposed on the simple interface
pressure_default = float(df["AirPressure"].median())
humidity_default = float(df["RelativeAirHumidity"].median())


@app.route("/")
def home():
    return render_template(
        "index.html",
        wind_min=wind_min,
        wind_max=wind_max,
        wind_default=round(wind_default, 1),
        temperature_min=temperature_min,
        temperature_max=temperature_max,
        temperature_default=round(temperature_default, 1)
    )


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        sunlight_percent = float(data["sunshine_percent"])
        temperature = float(data["temperature"])
        wind_speed = float(data["wind_speed"])

        # Convert sunlight percentage into the dataset's Sunshine range
        sunshine = np.interp(
            sunlight_percent,
            [0, 100],
            [df["Sunshine"].min(), df["Sunshine"].max()]
        )

        # Convert sunlight percentage into the dataset's Radiation range
        radiation = np.interp(
            sunlight_percent,
            [0, 100],
            [df["Radiation"].min(), df["Radiation"].max()]
        )

        # Automatically use current date and time
        now = datetime.now()

        hour = now.hour
        day = now.day
        month = now.month
        day_of_week = now.weekday()
        day_of_year = now.timetuple().tm_yday

        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)

        day_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_cos = np.cos(2 * np.pi * day_of_year / 365)

        # Build model input
        input_data = pd.DataFrame([{
            "WindSpeed": wind_speed,
            "Sunshine": sunshine,
            "AirPressure": pressure_default,
            "Radiation": radiation,
            "AirTemperature": temperature,
            "RelativeAirHumidity": humidity_default,
            "HOUR": hour,
            "DAY": day,
            "MONTH": month,
            "DAY_OF_WEEK": day_of_week,
            "DAY_OF_YEAR": day_of_year,
            "HOUR_SIN": hour_sin,
            "HOUR_COS": hour_cos,
            "DAY_SIN": day_sin,
            "DAY_COS": day_cos
        }])

        # Predict solar production
        prediction = model.predict(input_data[features])[0]

        prediction = max(0, float(prediction))

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(debug=True)
