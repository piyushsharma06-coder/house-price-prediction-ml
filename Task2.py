# ==========================================================
# Task 2: House Price Prediction using Linear Regression
# Author: Shivans Sharma
# Description:
# This program predicts house prices using a Linear Regression
# model with the Kaggle House Prices dataset.
# ==========================================================

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ----------------------------------------------------------
# Step 1: Load Dataset
# ----------------------------------------------------------
print("=" * 60)
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("=" * 60)

try:
    data = pd.read_csv("train.csv")
    print("\nDataset loaded successfully!\n")
except FileNotFoundError:
    print("\nError: train.csv not found.")
    print("Please place train.csv in the same folder as Task2.py")
    exit()

# ----------------------------------------------------------
# Step 2: Display Dataset Information
# ----------------------------------------------------------
print("First 5 Rows:")
print(data.head())

print("\nDataset Shape:")
print(data.shape)

print("\nMissing Values:")
print(data.isnull().sum().sort_values(ascending=False).head(10))

# ----------------------------------------------------------
# Step 3: Select Important Features
# ----------------------------------------------------------
features = [
    "GrLivArea",
    "BedroomAbvGr",
    "FullBath",
    "OverallQual",
    "GarageCars",
    "YearBuilt"
]

target = "SalePrice"

# ----------------------------------------------------------
# Step 4: Remove Missing Values
# ----------------------------------------------------------
dataset = data[features + [target]].dropna()

X = dataset[features]
y = dataset[target]

# ----------------------------------------------------------
# Step 5: Split Dataset
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# ----------------------------------------------------------
# Step 6: Train Linear Regression Model
# ----------------------------------------------------------
model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel trained successfully!")

# ----------------------------------------------------------
# Step 7: Predict House Prices
# ----------------------------------------------------------
predictions = model.predict(X_test)

# ----------------------------------------------------------
# Step 8: Evaluate Model
# ----------------------------------------------------------
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, predictions)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error : {mae:.2f}")
print(f"Mean Squared Error  : {mse:.2f}")
print(f"Root Mean Squared Error : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# ----------------------------------------------------------
# Step 9: Regression Coefficients
# ----------------------------------------------------------
print("\nFeature Importance")

coefficients = pd.DataFrame({
    "Feature": features,
    "Coefficient": model.coef_
})

print(coefficients)

# ----------------------------------------------------------
# Step 10: Correlation Heatmap
# ----------------------------------------------------------
plt.figure(figsize=(8,6))
sns.heatmap(dataset.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 11: Actual vs Predicted Plot
# ----------------------------------------------------------
plt.figure(figsize=(7,6))
plt.scatter(y_test, predictions)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 12: Prediction Error Plot
# ----------------------------------------------------------
errors = y_test - predictions

plt.figure(figsize=(7,5))
plt.hist(errors, bins=30)

plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

# ----------------------------------------------------------
# Step 13: Sample Predictions
# ----------------------------------------------------------
comparison = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": predictions.astype(int)
})

print("\nSample Predictions")
print(comparison.head(10))

print("\nTask Completed Successfully!")