import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("AQI MODEL ACCURACY CHECKER")
print("=" * 60)

# --------------------------------
# 1. Load Dataset
# --------------------------------
print("\n[1] Loading dataset...")
df = pd.read_csv("dataset/aqi_dataset.csv")

# Keep only required columns
required_cols = ['PM2.5', 'PM10', 'NO2', 'CO', 'O3', 'AQI']
df = df[required_cols]

# Remove missing values
df = df.dropna()

# Remove invalid AQI values (EPA max is 500)
df = df[df['AQI'] <= 500]
df = df[df['AQI'] > 0]

# Rename columns
df.columns = ['pm2_5', 'pm10', 'no2', 'co', 'o3', 'aqi']

print(f"✓ Dataset loaded: {len(df)} valid samples")

# Show sample
print(f"\nSample data:")
print(df.head())

# --------------------------------
# 2. Prepare Features
# --------------------------------
print("\n[2] Preparing features...")
X = df[['pm2_5', 'pm10', 'no2', 'co', 'o3']]
y = df['aqi']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"✓ Train: {len(X_train)} samples, Test: {len(X_test)} samples")

# --------------------------------
# 3. Load Trained Models
# --------------------------------
print("\n[3] Loading trained models...")

try:
    # Load individual models
    linear = joblib.load("models/linear_model.pkl")
    rf = joblib.load("models/rf_model.pkl")
    svr = joblib.load("models/svr_model.pkl")
    
    # Load hybrid model
    hybrid = joblib.load("models/hybrid_model.pkl")
    rf_model = hybrid["rf"]
    svr_model = hybrid["svr"]
    rf_weight = hybrid.get("rf_weight", 0.8)
    svr_weight = hybrid.get("svr_weight", 0.2)
    
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"✗ Error loading models: {e}")
    print("Please run train_models.py first to train the models.")
    exit()

# --------------------------------
# 4. Evaluate Each Model
# --------------------------------
print("\n" + "=" * 60)
print("MODEL ACCURACY RESULTS")
print("=" * 60)

def evaluate_model(name, model, X_test, y_test, is_hybrid=False, rf_w=0, svr_w=0):
    """Evaluate a single model and print metrics."""
    
    if is_hybrid:
        # Hybrid prediction
        rf_pred = rf_model.predict(X_test)
        svr_pred = svr_model.predict(X_test)
        predictions = (rf_w * rf_pred) + (svr_w * svr_pred)
    else:
        predictions = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    
    # Calculate accuracy within ±10, ±20, ±50 AQI points
    within_10 = np.mean(np.abs(predictions - y_test) <= 10) * 100
    within_20 = np.mean(np.abs(predictions - y_test) <= 20) * 100
    within_50 = np.mean(np.abs(predictions - y_test) <= 50) * 100
    
    print(f"\n{'-' * 50}")
    print(f"📊 {name}")
    print(f"{'-' * 50}")
    print(f"  MAE (Mean Absolute Error):     {mae:.2f} AQI points")
    print(f"  RMSE (Root Mean Square Error):   {rmse:.2f} AQI points")
    print(f"  R² Score (0-1, higher=better): {r2:.4f}")
    print(f"  Accuracy within ±10 AQI:       {within_10:.1f}%")
    print(f"  Accuracy within ±20 AQI:       {within_20:.1f}%")
    print(f"  Accuracy within ±50 AQI:       {within_50:.1f}%")
    
    # Show some example predictions
    print(f"\n  Sample Predictions (Actual → Predicted):")
    for i in range(min(5, len(y_test))):
        actual = y_test.iloc[i]
        pred = int(predictions[i])
        diff = abs(pred - actual)
        status = "✓" if diff <= 20 else "✗"
        print(f"    {status} {actual} → {pred} (off by {diff})")
    
    return mae, rmse, r2

# Evaluate Linear Regression
mae_lr, rmse_lr, r2_lr = evaluate_model("Linear Regression", linear, X_test, y_test)

# Evaluate Random Forest
mae_rf, rmse_rf, r2_rf = evaluate_model("Random Forest", rf, X_test, y_test)

# Evaluate SVR
mae_svr, rmse_svr, r2_svr = evaluate_model("SVR", svr, X_test, y_test)

# Evaluate Hybrid
mae_hybrid, rmse_hybrid, r2_hybrid = evaluate_model(
    f"Hybrid (RF {rf_weight:.0%} + SVR {svr_weight:.0%})", 
    None, X_test, y_test, 
    is_hybrid=True, 
    rf_w=rf_weight, 
    svr_w=svr_weight
)

# --------------------------------
# 5. Summary Comparison
# --------------------------------
print("\n" + "=" * 60)
print("SUMMARY: BEST MODEL")
print("=" * 60)

results = {
    "Linear Regression": (mae_lr, r2_lr),
    "Random Forest": (mae_rf, r2_rf),
    "SVR": (mae_svr, r2_svr),
    "Hybrid": (mae_hybrid, r2_hybrid),
}

# Sort by MAE (lower is better)
sorted_results = sorted(results.items(), key=lambda x: x[1][0])

print(f"\n{'Model':<<20} {'MAE':<<10} {'R²':<<10}")
print("-" * 40)
for name, (mae, r2) in sorted_results:
    marker = "⭐" if name == sorted_results[0][0] else "  "
    print(f"{marker} {name:<18} {mae:<10.2f} {r2:<10.4f}")

best_model = sorted_results[0][0]
print(f"\n🏆 Best Model: {best_model}")
print(f"   Expected accuracy: ±{sorted_results[0][1][0]:.0f} AQI points")

# --------------------------------
# 6. Feature Importance (Random Forest)
# --------------------------------
print("\n" + "=" * 60)
print("FEATURE IMPORTANCE")
print("=" * 60)
print("\nWhich pollutant matters most for AQI prediction?")
print("(From Random Forest model)\n")

importance = pd.DataFrame({
    'Pollutant': X.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

for _, row in importance.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"  {row['Pollutant']:<10} {row['Importance']:.4f} {bar}")

print("\n" + "=" * 60)
print("Done! Use these metrics to judge model quality.")
print("=" * 60)