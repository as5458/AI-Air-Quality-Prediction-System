import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error


# --------------------------------
# Load Dataset
# --------------------------------
data = pd.read_csv("dataset/aqi_dataset.csv")

# Keep required columns
data = data[["PM2.5","PM10","NO2","CO","O3","AQI"]]

# Remove missing values
data = data.dropna()

# Rename columns
data.columns = ["pm2_5","pm10","no2","co","o3","aqi"]

print("Dataset loaded successfully")
print("Total samples:", len(data))


# --------------------------------
# Features and Target
# --------------------------------
X = data[["pm2_5","pm10","no2","co","o3"]]
y = data["aqi"]


# --------------------------------
# Train Test Split
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# --------------------------------
# Evaluation Function
# --------------------------------
def evaluate(name, y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{name} Results")
    print("MAE :", round(mae,3))
    print("RMSE:", round(rmse,3))


# --------------------------------
# Linear Regression
# --------------------------------
linear = LinearRegression()
linear.fit(X_train, y_train)

pred_lr = linear.predict(X_test)

evaluate("Linear Regression", y_test, pred_lr)

joblib.dump(linear, "models/linear_model.pkl")


# --------------------------------
# Random Forest
# --------------------------------
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42
)

rf.fit(X_train, y_train)

pred_rf = rf.predict(X_test)

evaluate("Random Forest", y_test, pred_rf)

joblib.dump(rf, "models/rf_model.pkl")


# --------------------------------
# Support Vector Regression
# --------------------------------
svr = SVR(kernel="rbf")

svr.fit(X_train, y_train)

pred_svr = svr.predict(X_test)

evaluate("SVR", y_test, pred_svr)

joblib.dump(svr, "models/svr_model.pkl")


# --------------------------------
# Hybrid Model (RF + SVR)
# --------------------------------
rf_pred = rf.predict(X_test)
svr_pred = svr.predict(X_test)

# Weighted ensemble
hybrid_pred = (0.6 * rf_pred) + (0.4 * svr_pred)

evaluate("Hybrid (RF + SVR)", y_test, hybrid_pred)

# Save hybrid models
joblib.dump(
    {"rf": rf, "svr": svr},
    "models/hybrid_model.pkl"
)

print("\nAll models trained and saved successfully.")