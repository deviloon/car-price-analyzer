import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.linear_model import Ridge
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.compose import TransformedTargetRegressor
from sklearn.model_selection import GridSearchCV, KFold
import copy
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
import xgboost as xgb
import pyarrow
import phik
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import catboost
from catboost import CatBoostRegressor
import mlflow.sklearn
from phik.report import plot_correlation_matrix
import mlflow
import time
import os
from datetime import datetime
import category_encoders as ce
import joblib



# mlflow ui --backend-store-uri sqlite:///C:/project/car-price-analyzer/src/mlflow_runs/mlflow.db


# Создадим словарь конфигураций
CONFIG = {
    # Константы
    "DEV_MODE": False,
    "DEV_SAMPLE_SIZE": 100000,
    "RANDOM_STATE": 42,
    # Целевая переменная 
    "TARGET": "Цена",
    "YEAR": datetime.now().year
}

train = pd.read_parquet("../data/features/train_features.parquet")
test = pd.read_parquet("../data/features/test_features.parquet")

y_train = train[CONFIG["TARGET"]]
X_train = train.drop(columns=[CONFIG["TARGET"]])

y_test = test[CONFIG["TARGET"]]
X_test = test.drop(columns=[CONFIG["TARGET"]])

dir = 'C:/project/car-price-analyzer/src/mlflow_runs'
os.makedirs(dir, exist_ok=True)
mlflow.set_tracking_uri(f'sqlite:///{dir}/mlflow.db')
mlflow.set_experiment('HPO_benchmark')

cv = KFold(n_splits=5, shuffle=True, random_state=CONFIG['RANDOM_STATE'])

model = lgb.LGBMRegressor(
    objective='mape',
    random_state=CONFIG['RANDOM_STATE'],
    n_jobs=-1,
    verbose=-1
)

param_grid = {
    'learning_rate': [0.01, 0.03, 0.08, 0.15],
    'num_leaves': [31, 63, 127, 255],
    'min_child_samples': [20, 50, 150],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],              
    'n_estimators': [500]
    }

grid_search = GridSearchCV(
    estimator = model,
    param_grid=param_grid,
    cv=cv,
    scoring='neg_mean_absolute_percentage_error',
    n_jobs=-1,
    verbose=1
)

run_name='gridsearch'
with mlflow.start_run(run_name=run_name):
    print('Запуск GridSearchCV...')
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    total_time = time.time() - start_time
    best_mape = -grid_search.best_score_
    best_params = -grid_search.best_params_

    mlflow.log_params(best_params)
    mlflow.log_metric('best_mape', best_mape)
    mlflow.log_metric('total_time_sec', total_time)

    running_best = float('inf')
    for i, score in enumerate(grid_search.cv_results_['mean_test_score']):
        current_mape = -score
        running_best = min(running_best, current_mape)
        
        mlflow.log_metric('convergence_mape', running_best, step=i)

    print(f"GridSearch завершен за {total_time / 60:.2f} минут. Лучший MAPE: {best_mape:.4f}")

    