
# =====================================================
# SOLARPREDICT AI
# STEP 1: DATASET LOADING AND EDA
# =====================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Create plots folder
os.makedirs("plots", exist_ok=True)

# 2. Load dataset
file_path = "dataset/Solar Power Plant Data.csv"

df = pd.read_csv(file_path)

print("\n========== DATASET LOADED ==========")
print("Dataset loaded successfully!")

# 3. Display dataset information
print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)

print("\n========== DATASET INFORMATION ==========")
df.info()

# 4. Statistical description
print("\n========== STATISTICAL SUMMARY ==========")
print(df.describe())

# 5. Check missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Check duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# 7. Convert date column
date_column = "Date-Hour(NMT)"

df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce",
    dayfirst=True
)

print("\n========== DATE CONVERSION ==========")
print(df[date_column].head())

print("Invalid dates:", df[date_column].isnull().sum())

# 8. Check target column
target = "SystemProduction"

print("\n========== TARGET ANALYSIS ==========")
print(df[target].describe())

# 9. Create correlation heatmap
plt.figure(figsize=(10, 7))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Solar Dataset Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png")
plt.show()

# 10. Solar production distribution
plt.figure(figsize=(8, 5))

sns.histplot(
    df[target].dropna(),
    bins=40,
    kde=True
)

plt.title("Solar Energy Production Distribution")
plt.xlabel("System Production")
plt.ylabel("Frequency")

plt.tight_layout()
plt.savefig("plots/production_distribution.png")
plt.show()

# 11. Solar radiation vs production
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Radiation",
    y=target,
    alpha=0.5
)

plt.title("Solar Radiation vs System Production")
plt.tight_layout()
plt.savefig("plots/radiation_vs_production.png")
plt.show()

# 12. Temperature vs production
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="AirTemperature",
    y=target,
    alpha=0.5
)

plt.title("Air Temperature vs System Production")
plt.tight_layout()
plt.savefig("plots/temperature_vs_production.png")
plt.show()

# 13. Sunshine vs production
plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=df,
    x="Sunshine",
    y=target,
    alpha=0.5
)

plt.title("Sunshine vs System Production")
plt.tight_layout()
plt.savefig("plots/sunshine_vs_production.png")
plt.show()

# 14. Production over time
time_df = df.sort_values(date_column)

plt.figure(figsize=(12, 5))

plt.plot(
    time_df[date_column],
    time_df[target]
)

plt.title("Solar Energy Production Over Time")
plt.xlabel("Date")
plt.ylabel("System Production")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("plots/production_over_time.png")
plt.show()

print("\n========== EDA COMPLETED ==========")
print("All EDA graphs saved inside the plots folder.")


# =====================================================
# STEP 2: DATA CLEANING
# =====================================================

print("\n========== DATA CLEANING ==========")

# Create a copy of original dataset
clean_df = df.copy()

# Remove duplicate records
clean_df = clean_df.drop_duplicates()

# Convert input columns to numeric
numeric_columns = [
    "WindSpeed",
    "Sunshine",
    "AirPressure",
    "Radiation",
    "AirTemperature",
    "RelativeAirHumidity",
    "SystemProduction"
]

for col in numeric_columns:
    clean_df[col] = pd.to_numeric(
        clean_df[col], errors="coerce"
    )

# Remove records with invalid dates or missing target
clean_df = clean_df.dropna(
    subset=["Date-Hour(NMT)", "SystemProduction"]
)

# Handle missing input values using median
for col in numeric_columns[:-1]:
    clean_df[col] = clean_df[col].fillna(
        clean_df[col].median()
    )

# Sort records chronologically
clean_df = clean_df.sort_values(
    "Date-Hour(NMT)"
).reset_index(drop=True)

print("Original dataset shape:", df.shape)
print("Cleaned dataset shape:", clean_df.shape)

print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())

print("\nNegative production records:")
print((clean_df["SystemProduction"] < 0).sum())

print("\nNegative radiation records:")
print((clean_df["Radiation"] < 0).sum())

print("\nData cleaning completed!")


# =====================================================
# STEP 3: FEATURE ENGINEERING
# =====================================================

print("\n========== FEATURE ENGINEERING ==========")

# Extract time features
clean_df["HOUR"] = clean_df["Date-Hour(NMT)"].dt.hour

clean_df["DAY"] = clean_df["Date-Hour(NMT)"].dt.day

clean_df["MONTH"] = clean_df["Date-Hour(NMT)"].dt.month

clean_df["DAY_OF_WEEK"] = (
    clean_df["Date-Hour(NMT)"].dt.dayofweek
)

clean_df["DAY_OF_YEAR"] = (
    clean_df["Date-Hour(NMT)"].dt.dayofyear
)

# Cyclical encoding for hour
clean_df["HOUR_SIN"] = np.sin(
    2 * np.pi * clean_df["HOUR"] / 24
)

clean_df["HOUR_COS"] = np.cos(
    2 * np.pi * clean_df["HOUR"] / 24
)

# Cyclical encoding for day of year
clean_df["DAY_SIN"] = np.sin(
    2 * np.pi * clean_df["DAY_OF_YEAR"] / 365.25
)

clean_df["DAY_COS"] = np.cos(
    2 * np.pi * clean_df["DAY_OF_YEAR"] / 365.25
)

print("\nNew features created:")
print(clean_df.columns.tolist())

# Save processed dataset
clean_df.to_csv(
    "dataset/solar_features.csv",
    index=False
)

print("\nProcessed dataset saved successfully!")
print("File: dataset/solar_features.csv")
print("\n========== PREPROCESSING COMPLETED ==========")


# =====================================================
# STEP 4: TRAIN-TEST SPLIT
# =====================================================

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib

print("\n========== TRAIN TEST SPLIT ==========")

# Select input features
features = [
    "WindSpeed",
    "Sunshine",
    "AirPressure",
    "Radiation",
    "AirTemperature",
    "RelativeAirHumidity",
    "HOUR",
    "DAY",
    "MONTH",
    "DAY_OF_WEEK",
    "DAY_OF_YEAR",
    "HOUR_SIN",
    "HOUR_COS",
    "DAY_SIN",
    "DAY_COS"
]

X = clean_df[features]
y = clean_df["SystemProduction"]


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\n========== RANDOM SPLIT ==========")
print("Training records:", len(X_train))
print("Testing records:", len(X_test))
print("Training average:", round(y_train.mean(), 2))
print("Testing average:", round(y_test.mean(), 2))

# =====================================================
# STEP 5: MODEL TRAINING
# =====================================================

print("\n========== MODEL TRAINING ==========")

models = {
    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        min_samples_split=5,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=42,
        n_jobs=-1
    )
}

results = []
trained_models = {}

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)

    mse = mean_squared_error(y_test, predictions)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, predictions)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = model

    print(name, "training completed!")

    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2 Score:", round(r2, 4))


# =====================================================
# STEP 6: MODEL COMPARISON
# =====================================================

print("\n========== MODEL COMPARISON ==========")

results_df = pd.DataFrame(results)

print(results_df.to_string(index=False))

# Select best model using lowest RMSE
best_model_name = results_df.loc[
    results_df["RMSE"].idxmin(), "Model"
]

best_model = trained_models[best_model_name]

print("\nBest model:", best_model_name)

# Save models
os.makedirs("models", exist_ok=True)

for name, model in trained_models.items():

    filename = name.lower().replace(" ", "_") + "_model.pkl"

    joblib.dump(model, f"models/{filename}")

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

# Save feature names
joblib.dump(features, "models/features.pkl")

# Save comparison results
results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("\nAll models saved successfully!")
print("Best model saved as models/best_model.pkl")

accuracy = max(0, r2 * 100)

print("Accuracy:", round(accuracy, 2), "%")

print("\n========== PRODUCTION ANALYSIS ==========")

print("Minimum production:", clean_df["SystemProduction"].min())
print("Maximum production:", clean_df["SystemProduction"].max())
print("Average production:", clean_df["SystemProduction"].mean())

print(
    "Zero production records:",
    (clean_df["SystemProduction"] == 0).sum()
)

print(
    "Negative production records:",
    (clean_df["SystemProduction"] < 0).sum()
)

print("\nProduction by month:")

print(
    clean_df.groupby("MONTH")["SystemProduction"].mean()
)


print("\n========== DAYTIME PRODUCTION ANALYSIS ==========")

daytime_df = clean_df[clean_df["SystemProduction"] > 0]

print("Daytime records:", len(daytime_df))
print("Daytime average production:", daytime_df["SystemProduction"].mean())
print("Daytime minimum production:", daytime_df["SystemProduction"].min())
print("Daytime maximum production:", daytime_df["SystemProduction"].max())

print("\nPositive production records by month:")
print(daytime_df.groupby("MONTH")["SystemProduction"].count())

print("\nZero production percentage:")
zero_percentage = (
    (clean_df["SystemProduction"] == 0).sum() / len(clean_df)
) * 100

print(round(zero_percentage, 2), "%")


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Predict using the already trained XGBoost model
xgb_model = models["XGBoost"]

daytime_mask = y_test > 0

X_daytime = X_test.loc[daytime_mask]
y_daytime = y_test.loc[daytime_mask]

daytime_predictions = xgb_model.predict(X_daytime)

daytime_mae = mean_absolute_error(y_daytime, daytime_predictions)
daytime_rmse = np.sqrt(mean_squared_error(y_daytime, daytime_predictions))
daytime_r2 = r2_score(y_daytime, daytime_predictions)

print("\n========== DAYTIME XGBOOST PERFORMANCE ==========")
print("Daytime test records:", len(y_daytime))
print("MAE:", round(daytime_mae, 2))
print("RMSE:", round(daytime_rmse, 2))
print("R2 Score:", round(daytime_r2, 4))
print("R2 Percentage:", round(daytime_r2 * 100, 2), "%")


print("\n========== TRAINING VS TESTING ANALYSIS ==========")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ACTUAL VS PREDICTED GRAPH

import matplotlib.pyplot as plt
import os

xgb_model = models["XGBoost"]
predictions = xgb_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.4)

plt.xlabel("Actual Solar Production")
plt.ylabel("Predicted Solar Production")
plt.title("XGBoost - Actual vs Predicted")

plt.tight_layout()
plt.savefig("plots/actual_vs_predicted.png")
plt.show()

print("Actual vs Predicted graph saved!")


# FEATURE IMPORTANCE ANALYSIS

import pandas as pd
import matplotlib.pyplot as plt

xgb_model = models["XGBoost"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": xgb_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance", ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance)

plt.figure(figsize=(10, 6))

plt.barh(
    importance["Feature"],
    importance["Importance"]
)

plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.title("XGBoost Feature Importance")

plt.gca().invert_yaxis()
plt.tight_layout()

plt.savefig("plots/feature_importance.png")
plt.show()

print("Feature importance graph saved!")
