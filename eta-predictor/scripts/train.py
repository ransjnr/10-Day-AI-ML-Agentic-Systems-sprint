import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
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

    if 'vehicle_type' in df.columns:
        df['vehicle_van'] = (df['vehicle_type'] == 'van').astype(float)
        df['vehicle_truck'] = (df['vehicle_type'] == 'truck').astype(float)
        df['vehicle_motorcycle'] = (df['vehicle_type'] == 'motorcycle').astype(float)
    else:
        df['vehicle_van'] = 0.0
        df['vehicle_truck'] = 0.0
        df['vehicle_motorcycle'] = 0.0

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
    data_path: str,
        save_model: bool = False,
) -> dict:

    # Connect to MLflow tracking server
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    # Load and prepare data
    X_train, X_val, y_train, y_val = load_and_prepare(data_path)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n{'='*60}")
        print(f'Run: {run_name}')
        print(f'MLflow Run ID: {run.info.run_id}')

        mlflow.set_tags({
            'model_type': model_type,
            'dataset': 'logistics-ghana-v1',
            'num-features': len(FEATURES_COLS),
            'features': ','.join(FEATURES_COLS),
        })

        # === 2. Log all hyperparameters ===
        mlflow.log_param('model_type', model_type)
        mlflow.log_param('train_samples', len(X_train))
        mlflow.log_param('val_samples', len(X_val))
        mlflow.log_params(hyperparams)


        # === 3. Build model pipeline ===
        if model_type == 'gbr':
            regressor = GradientBoostingRegressor(**hyperparams, random_state=42)
        elif model_type == 'rf':
            regressor = RandomForestRegressor(**hyperparams, random_state=42)
        elif model_type == 'ridge':
            regressor = Ridge(**hyperparams)
        else:
            raise ValueError(f'Unsupported model type: {model_type}')
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', regressor)
        ])

        # === 4. Train the model ===
        print('Training model...')
        pipeline.fit(X_train, y_train)

        # === 5. Cross-validation (5 folds) ===
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5,
            scoring='neg_mean_absolute_error',
            n_jobs=-1
        )
        cv_mae = -cv_scores.mean()
        cv_std = cv_scores.std()
        mlflow.log_metric('cv_mae_mean', round(cv_mae, 4))
        mlflow.log_metric('cv_mae_std', round(cv_std, 4))

        # === 6. Evaluate on validation set ===
        y_pred = pipeline.predict(X_val)

        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / (y_val + 1e-8))) * 100

        metrics = {
            'val_mae': round(mae, 4),
            'val_rmse': round(rmse, 4),
            'val_r2': round(r2, 4),
            'val_mape': round(mape, 4)
        }
        mlflow.log_metrics(metrics)

        print(f' val_mae: {mae:.2f} minutes')
        print(f' val_rmse: {rmse:.2f} minutes')
        print(f' val_r2: {r2:.4f}')
        print(f' cv_mae: {cv_mae:.2f} minutes (std: {cv_std:.2f})')
              
        # === 7. Save plots as MLflow artifacts ===
        Path('mlflow_plots').mkdir(exist_ok=True)
        fi_path = plot_feature_importance(pipeline, FEATURES_COLS, 'mlflow_plots/feature_importance.png')
        if fi_path:
            mlflow.log_artifact(fi_path, artifact_path='plots')
        avp_path = plot_actual_vs_predicted(y_val, y_pred, 'mlflow_plots/actual_vs_predicted.png', mae)
        mlflow.log_artifact(avp_path, artifact_path='plots')

        # === 8. Log the trained model ===
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path='model',
            registered_model_name='eta-predictor',

            input_example=X_val[:5].tolist(),
        )

        # === 9. Optionally save the model locally ===
        if save_model:
            Path('models').mkdir(exist_ok=True)
            model_path = f'models/eta_model_{run.info.run_id[:8]}.joblib'
            joblib.dump(pipeline, model_path)

            joblib.dump(pipeline, 'models/eta_model_latest.joblib')
            print(f'Model saved to {model_path}')

            mlflow.log_artifact(model_path, artifact_path='model')
        
        print(f'  MLflow UI: http://localhost:5000)')
        return metrics

if __name__ == '__main__':
    DATA_PATH = 'data/raw/logistics_eta.csv'

    print('Starting training experiments...')
    print('View live at MLflow UI: http://localhost:5000')


    # === Experiment 1: Gradient Boosting Regressor (Baseline) ===
    train_and_log(
        run_name='gbr_baseline',
        model_type='gbr',
        hyperparams={
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 3
        },
        data_path=DATA_PATH,
        save_model=True
    )

    # Experiment 2: More tress - does it improve?
    train_and_log(
        run_name='gbr_more_trees',
        model_type='gbr',
        hyperparams={
            'n_estimators': 500,
            'learning_rate': 0.03,
            'max_depth': 4
        },
        data_path=DATA_PATH
    )

    # Experiment 3: Deeper trees
    train_and_log(
        run_name='gbr_deeper_trees',
        model_type='gbr',
        hyperparams={
            'n_estimators': 200,
            'learning_rate': 0.05,
            'max_depth': 6
        },
        data_path=DATA_PATH
    )

    # Experiment 4: Random Forest Comparison
    train_and_log(
        run_name='rf_comparison',
        model_type='rf',
        hyperparams={
            'n_estimators': 200,
            'max_depth': 10
        },
        data_path=DATA_PATH
    )

    # Experiment 5: Linear baseline 
    train_and_log(
        run_name='ridge_comparison',
        model_type='ridge',
        hyperparams={
            'alpha': 1.0
        },
        data_path=DATA_PATH
    )

    print('All experiments completed! Check the MLflow UI for details.')
    print('Open http://localhost:5000 in your browser to explore the results.')
    print('Look for the run with the best validation MAE and click through to see the feature importance and actual vs predicted plots.')