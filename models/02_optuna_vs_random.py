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
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
import copy
from sklearn.model_selection import cross_val_score
import optuna
from sklearn.model_selection import cross_validate
import xgboost as xgb
import pyarrow
import phik
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
import catboost
from catboost import CatBoostRegressor
import mlflow.sklearn
from phik.report import plot_correlation_matrix
import mlflow
import time
from scipy.stats import randint, uniform, loguniform
import os
from datetime import datetime
import category_encoders as ce
import joblib



# mlflow ui --backend-store-uri sqlite:///C:/project/car-price-analyzer/src/mlflow_runs/mlflow.db


# Создадим словарь конфигураций
CONFIG = {
    # Константы
    "DEV_MODE": False,
    "DEV_SAMPLE_SIZE": 2000,
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
    objective='mae',
    random_state=CONFIG['RANDOM_STATE'],
    n_jobs=1,
    verbose=-1,
    subsample_freq=1
)

wrapped_model = TransformedTargetRegressor(
    regressor=model,
    func=np.log1p,
    inverse_func=np.expm1
)

param_grid = {
    'learning_rate': [0.01, 0.03, 0.08, 0.15],
    'num_leaves': [31, 63, 127, 255],
    'min_child_samples': [20, 50, 150],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],              
    'n_estimators': [500]
}

param_distributions = {
    'learning_rate': loguniform(0.01, 0.15),
    'num_leaves': randint(31, 256),
    'min_child_samples': randint(20, 151),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'n_estimators': [500]
}

scoring = {
    'mape': 'neg_mean_absolute_percentage_error',
    'mae': 'neg_mean_absolute_error'
}

param_grid_wrapped = {
    f'regressor__{key}': value for key, value in param_grid.items()
}

param_distributions_wrapped = {
    f'regressor__{key}': value for key, value in param_distributions.items()
}

grid_search = GridSearchCV(
    estimator = wrapped_model,
    param_grid=param_grid_wrapped,
    cv=cv,
    scoring=scoring,
    refit='mape',
    n_jobs=-1,
    verbose=2
)

random_search = RandomizedSearchCV(
    estimator=wrapped_model,
    param_distributions=param_distributions_wrapped,
    scoring=scoring,
    cv=cv,
    refit='mape',
    n_jobs=-1,
    verbose=2,
    random_state=CONFIG['RANDOM_STATE'],
    n_iter = 60
)

# run_name='gridsearch' # mae = 139653.7310652461
# with mlflow.start_run(run_name=run_name):
#     if CONFIG["DEV_MODE"]:
#         print(f"Включен DEV_MODE. Обучение на {CONFIG['DEV_SAMPLE_SIZE']} строках...")
#         train = train.sample(n=min(CONFIG["DEV_SAMPLE_SIZE"], len(train)), random_state=CONFIG["RANDOM_STATE"])
#         y_train = train[CONFIG["TARGET"]]
#         X_train = train.drop(columns=[CONFIG["TARGET"]])

#     print('Запуск GridSearchCV...')
#     start_time = time.time()
#     grid_search.fit(X_train, y_train)
#     total_time = time.time() - start_time
#     best_idx = grid_search.best_index_

#     best_mape = -grid_search.cv_results_['mean_test_mape'][best_idx]
#     best_mae = -grid_search.cv_results_['mean_test_mae'][best_idx]
#     best_params = grid_search.best_params_
#     mlflow.log_params(best_params)
#     mlflow.log_metric('best_mape', best_mape)
#     mlflow.log_metric('best_mae', best_mae)
#     mlflow.log_metric('total_time_sec', total_time)


#     running_best_mape = float('inf')
#     running_best_mae = float('inf')

#     num_combinations = len(grid_search.cv_results_['params'])
#     for i in range(num_combinations):
#         # Достаем метрики конкретной i-й итерации
#         current_mape = -grid_search.cv_results_['mean_test_mape'][i]
#         current_mae = -grid_search.cv_results_['mean_test_mae'][i]

#         # Обновляем лучшие нарастающие значения
#         running_best_mape = min(running_best_mape, current_mape)
#         running_best_mae = min(running_best_mae, current_mae)

#         mlflow.log_metric('convergence_mape', running_best_mape, step=i)
#         mlflow.log_metric('convergence_mae', running_best_mae, step=i)

#     print(f"GridSearch завершен за {total_time / 60:.2f} минут.")
#     print(f"Лучший MAPE: {best_mape:.4f} | MAE этой же модели: {best_mae:.2f}")



# run_name='random_search'
# with mlflow.start_run(run_name=run_name):
#     if CONFIG["DEV_MODE"]:
#         print(f"Включен DEV_MODE. Обучение на {CONFIG['DEV_SAMPLE_SIZE']} строках...")
#         train = train.sample(n=min(CONFIG["DEV_SAMPLE_SIZE"], len(train)), random_state=CONFIG["RANDOM_STATE"])
#         y_train = train[CONFIG["TARGET"]]
#         X_train = train.drop(columns=[CONFIG["TARGET"]])

#     print('Запуск RandomizedSearchCV...')
#     start_time = time.time()
#     random_search.fit(X_train, y_train)
#     total_time = time.time() - start_time
#     best_idx = random_search.best_index_

#     best_mape = -random_search.cv_results_['mean_test_mape'][best_idx]
#     best_mae = -random_search.cv_results_['mean_test_mae'][best_idx]
#     best_params = random_search.best_params_
#     mlflow.log_params(best_params)
#     mlflow.log_metric('best_mape', best_mape)
#     mlflow.log_metric('best_mae', best_mae)
#     mlflow.log_metric('total_time_sec', total_time)


#     running_best_mape = float('inf')
#     running_best_mae = float('inf')

#     num_combinations = len(random_search.cv_results_['params'])
#     for i in range(num_combinations):
#         # Достаем метрики конкретной i-й итерации
#         current_mape = -random_search.cv_results_['mean_test_mape'][i]
#         current_mae = -random_search.cv_results_['mean_test_mae'][i]

#         # Обновляем лучшие нарастающие значения
#         running_best_mape = min(running_best_mape, current_mape)
#         running_best_mae = min(running_best_mae, current_mae)

#         mlflow.log_metric('convergence_mape', running_best_mape, step=i)
#         mlflow.log_metric('convergence_mae', running_best_mae, step=i)

#     print(f"RandomSearch завершен за {total_time / 60:.2f} минут.")
#     print(f"Лучший MAPE: {best_mape:.4f} | MAE этой же модели: {best_mae:.2f}")


run_name='optuna'
with mlflow.start_run(run_name=run_name):
    def objective(trial):
        lgb_params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 31, 255),
            'min_child_samples': trial.suggest_int('min_child_samples', 20, 150),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'n_estimators': 500,
            'objective': 'mae',
            'random_state': CONFIG['RANDOM_STATE'],
            'n_jobs': 1,
            'verbose': -1,
            'subsample_freq': 1
        }
        wrapped_model = TransformedTargetRegressor(
            regressor=model,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        scores = cross_validate(
            estimator=wrapped_model,
            X=X_train,
            y=y_train,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )
        
        mape = -scores['test_mape'].mean()
        mae = -scores['test_mae'].mean()
        
        trial.set_user_attr('mae', mae)
        
        return mape

    print('Запуск Optuna...')
    start_time = time.time()
    
    # Отключаем лишний мусор в консоли от Optuna, оставляя только важные сообщения
    optuna.logging.set_verbosity(optuna.logging.INFO)
    
    sampler = optuna.samplers.TPESampler(seed=CONFIG['RANDOM_STATE'])
    study = optuna.create_study(direction='minimize', sampler=sampler)
    
    study.optimize(objective, n_trials=60)
    
    total_time = time.time() - start_time
    
    best_mape = study.best_value
    best_mae = study.best_trial.user_attrs['mae']
    
    # Форматируем параметры для логгирования (добавляем 'regressor__' для идентичности с sklearn)
    best_params = {f'regressor__{k}': v for k, v in study.best_params.items()}
    best_params['regressor__n_estimators'] = 500
    
    mlflow.log_params(best_params)
    mlflow.log_metric('best_mape', best_mape)
    mlflow.log_metric('best_mae', best_mae)
    mlflow.log_metric('total_time_sec', total_time)

    # Строим график сходимости
    running_best_mape = float('inf')
    running_best_mae = float('inf')
    
    for i, trial in enumerate(study.trials):
        if trial.state != optuna.trial.TrialState.COMPLETE:
            continue
            
        current_mape = trial.value
        current_mae = trial.user_attrs.get('mae', float('inf'))
        running_best_mape = min(running_best_mape, current_mape)
        running_best_mae = min(running_best_mae, current_mae)

        mlflow.log_metric('convergence_mape', running_best_mape, step=i)
        mlflow.log_metric('convergence_mae', running_best_mae, step=i)

    print(f"Optuna завершена за {total_time / 60:.2f} минут.")
    print(f"Лучший MAPE: {best_mape:.4f} | MAE этой же модели: {best_mae:.2f}")