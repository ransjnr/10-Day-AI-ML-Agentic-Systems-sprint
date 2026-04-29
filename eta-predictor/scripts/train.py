import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error,  r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

# === Configuration ===

FEATURES_COLS = [
    'distance_km',
    'cargo_weight_kg',
    'is_rush_hour',
    'day_of_week',
    'num_stops',
    'hour_of_day',
    'traffic_index',
    'vehicle_van',
    'vehicle_truck',
    'vehicle_motorcycle'
]
TARGET_COL = 'eta_minutes'
MLFLOW_URI = 'http://localhost:5000'
EXPERIMENT_NAME = 'eta-predictor-day1' 

# === Load Data ===

def load_and_prepare (csv_path: str):
    df = pd.read_csv(csv_path)
    print(f'Loaded {len(df)} samples from {csv_path}')

    rush = list(range(7, 10)) + list(range(17, 20))
    df['is_rush_hour'] = df['hour_of_day'].isin(rush).astype(float)

    df['vehicle_van'] = (df['vehicle_type'] == 'van').astype(float)
    df['vehicle_truck'] = (df['vehicle_type'] == 'truck').astype(float)
    df['vehicle_motorcycle'] = (df['vehicle_type'] == 'motorcycle').astype(float)

    # Ensure all feature columns are present
    for col in FEATURES_COLS:
        if col not in df.columns:
            df[col] = 0.0

    X = df[FEATURES_COLS].values
    y = df[TARGET_COL].values

    # Split into train and test sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f'Train samples: {len(X_train)}, Validation samples: {len(X_val)}')
    return X_train, X_val, y_train, y_val

# === Visualisation ===

def plot_feature_importance(model_pipeline, feature_names: list, save_path: str):
    regressor = model_pipeline.named_steps['regressor']
    if not hasattr(regressor, 'feature_importances_'):
        return None
    
    importances = regressor.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(9,5))
    bars = ax.barh(
        [feature_names[i] for i in sorted_idx],
        importances[sorted_idx],
        color='#1565c0'
    )
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title('Which features matter most for predicting ETA', fontsize=12)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

def plot_actual_vs_predicted(y_val, y_pred, save_path: str, mae: float):
    fig, ax = plt.subplots(figsize=(7,7))
    ax.scatter(y_val, y_pred, alpha=0.3, s=8, color='#1565C0')

    min_val = min(min(y_val), min(y_pred))
    max_val = max(max(y_val), max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Predictions')
    ax.set_xlabel('Actual ETA (minutes)', fontsize=11)
    ax.set_ylabel('Predicted ETA (minutes)', fontsize=11)
    ax.set_title(f'Actual vs Predicted ETA (MAE: {mae:.2f} mins)', fontsize=12)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path

# == Main Training Function =============

def train_and_log(
        run_name: str,
        model_type: str,
        hyperparams: dict,
        data_path: str
        save_model: bool = False,
) -> dict:

    # Connect to MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and prepare data
    X_train, X_val, y_train, y_val = load_and_prepare(data_path)

    with mlflow.start_run(run_name=run_name) as run:
        print