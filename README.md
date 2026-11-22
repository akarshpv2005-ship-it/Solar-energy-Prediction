# ☀️ Solar Energy Prediction

Predict solar energy output using Machine Learning based on weather conditions like temperature, humidity, irradiance, wind speed, and more.

This project compares multiple ML models (Decision Tree, Random Forest, XGBoost) and serves the best model through a Flask web application with a clean, interactive UI.

## 🚀 Features

- Predict solar power generation in real-time
- Compare Decision Tree, Random Forest & XGBoost models
- Interactive web UI with instant predictions
- Data visualization with plots
- Trained models saved as `.pkl` files for fast inference
- Clean Flask backend + HTML/CSS/JS frontend

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **Backend:** Flask
- **ML Libraries:** Scikit-learn, XGBoost, Pandas, NumPy
- **Frontend:** HTML, CSS, JavaScript
- **Visualization:** Matplotlib / Seaborn
- **Models:** Decision Tree, Random Forest, XGBoost

# To run
pip install -r requirements.txt
python train_model.py
python app.py