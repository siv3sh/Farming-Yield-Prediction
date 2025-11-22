#!/usr/bin/env python3
"""
Standalone script to train and save the XGBoost model for the dashboard
"""
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb
import os

print("🌾 Training XGBoost Model for Dashboard...")
print("=" * 60)

# Load data
print("\n1. Loading dataset...")
df = pd.read_csv('Synthetic_Farming_Dataset_With_Seasonality_And_Challenge.csv')
print(f"   ✅ Loaded {len(df)} records")

# Handle missing values
print("\n2. Preprocessing data...")
df_processed = df.copy()
df_processed['soil_N'].fillna(df_processed['soil_N'].median(), inplace=True)
df_processed['rainfall_mm'].fillna(df_processed['rainfall_mm'].median(), inplace=True)
df_processed['fertilizer_kg_per_ha'].fillna(df_processed['fertilizer_kg_per_ha'].median(), inplace=True)

# Feature engineering
print("3. Engineering features...")
df_processed['soil_quality_index'] = (
    (df_processed['soil_N'] / df_processed['soil_N'].max()) * 0.4 +
    (df_processed['soil_P'] / df_processed['soil_P'].max()) * 0.3 +
    (df_processed['soil_pH'] / df_processed['soil_pH'].max()) * 0.3
) * 100

df_processed['input_intensity'] = (
    (df_processed['fertilizer_kg_per_ha'] / df_processed['fertilizer_kg_per_ha'].max()) * 0.4 +
    (df_processed['irrigation_mm'] / df_processed['irrigation_mm'].max()) * 0.4 +
    (df_processed['pesticide_ml'] / df_processed['pesticide_ml'].max()) * 0.2
) * 100

df_processed['season'] = df_processed['month'].apply(
    lambda x: 'Winter' if x in [12, 1, 2] else 
              'Spring' if x in [3, 4, 5] else 
              'Summer' if x in [6, 7, 8] else 'Autumn'
)

df_processed['rainfall_temp_ratio'] = df_processed['rainfall_mm'] / (df_processed['temp_avg'] + 1)
df_processed['fertilizer_efficiency'] = df_processed['yield_kg_per_ha'] / (df_processed['fertilizer_kg_per_ha'] + 1)

# Encode categorical variables
print("4. Encoding categorical variables...")
le_crop = LabelEncoder()
df_processed['crop_type_encoded'] = le_crop.fit_transform(df_processed['crop_type'])
df_processed = pd.get_dummies(df_processed, columns=['season'], prefix='season')

# Prepare features
feature_cols = [
    'soil_pH', 'soil_N', 'soil_P', 'rainfall_mm', 'temp_avg',
    'fertilizer_kg_per_ha', 'irrigation_mm', 'pesticide_ml',
    'soil_quality_index', 'input_intensity', 'rainfall_temp_ratio',
    'fertilizer_efficiency', 'crop_type_encoded',
    'season_Autumn', 'season_Spring', 'season_Summer', 'season_Winter'
]

X = df_processed[feature_cols].copy()
y = df_processed['yield_kg_per_ha'].copy()

# Split data
print("5. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=df_processed.loc[X.index]['crop_type']
)
print(f"   Training set: {len(X_train)} samples")
print(f"   Test set: {len(X_test)} samples")

# Scale features
print("6. Scaling features...")
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X_train.columns,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=X_test.columns,
    index=X_test.index
)

# Train XGBoost model
print("7. Training XGBoost model...")
xgb_model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train_scaled, y_train)

# Evaluate
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
y_pred = xgb_model.predict(X_test_scaled)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n8. Model Performance:")
print(f"   RMSE: {rmse:.2f}")
print(f"   R²: {r2:.4f}")
print(f"   MAE: {mae:.2f}")

# Save model and preprocessing objects
print("\n9. Saving model and preprocessing objects...")
try:
    with open('trained_xgboost_model.pkl', 'wb') as f:
        pickle.dump(xgb_model, f)
    print("   ✅ trained_xgboost_model.pkl")
    
    with open('feature_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("   ✅ feature_scaler.pkl")
    
    with open('label_encoder_crop.pkl', 'wb') as f:
        pickle.dump(le_crop, f)
    print("   ✅ label_encoder_crop.pkl")
    
    with open('feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    print("   ✅ feature_columns.pkl")
    
    # Verify file sizes
    print("\n10. Verifying saved files...")
    files = ['trained_xgboost_model.pkl', 'feature_scaler.pkl', 'label_encoder_crop.pkl', 'feature_columns.pkl']
    for file in files:
        size = os.path.getsize(file)
        if size > 0:
            print(f"   ✅ {file}: {size:,} bytes")
        else:
            print(f"   ❌ {file}: 0 bytes (ERROR!)")
    
    print("\n" + "=" * 60)
    print("✅ Model training and saving completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error saving model: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

