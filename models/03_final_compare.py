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
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
    plot_parallel_coordinate
)
from catboost import CatBoostRegressor
import mlflow.sklearn
from phik.report import plot_correlation_matrix
import plotly
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
mlflow.set_experiment('final_compare')

cv = KFold(n_splits=5, shuffle=True, random_state=CONFIG['RANDOM_STATE'])

scoring = {
    'mape': 'neg_mean_absolute_percentage_error',
    'mae': 'neg_mean_absolute_error'
}


# run_name = 'lightgbm_final'
# with mlflow.start_run(run_name=run_name):
#     print(f'Начало большого поиска Optuna для LightGBM...')
#     def objective(trial):
#         lgb_params = {
#             'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2, log=True),
#             'num_leaves': trial.suggest_int('num_leaves', 15, 512),
#             'min_child_samples': trial.suggest_int('min_child_samples', 10, 300),
#             'subsample': trial.suggest_float('subsample', 0.4, 1.0),
#             'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),
#             'n_estimators': trial.suggest_int('n_estimators', 400, 1500, step=100),
#             'objective': 'mae',
#             'random_state': CONFIG['RANDOM_STATE'],
#             'n_jobs': 1, 
#             'verbose': -1,
#             'subsample_freq': 1
#         }
        
#         # Создаем новую модель для каждого триала
#         current_model = lgb.LGBMRegressor(**lgb_params)
#         current_wrapped = TransformedTargetRegressor(
#             regressor=current_model,
#             func=np.log1p,
#             inverse_func=np.expm1
#         )
        
#         scores = cross_validate(
#             estimator=current_wrapped,
#             X=X_train,
#             y=y_train,
#             scoring=scoring,
#             cv=cv,
#             n_jobs=-1
#         )
        
#         mape = -scores['test_mape'].mean()
#         mae = -scores['test_mae'].mean()
#         trial.set_user_attr('mae', mae)
        
#         return mape



#     start_time = time.time()
#     optuna.logging.set_verbosity(optuna.logging.INFO)
#     sampler = optuna.samplers.TPESampler(seed=CONFIG['RANDOM_STATE'])
#     study = optuna.create_study(direction='minimize', sampler=sampler)
#     study.optimize(objective, n_trials=150)
#     time_elapsed = time.time() - start_time
#     print(f"Поиск завершен за {time_elapsed / 60:.2f} минут.")

#     best_cv_mape = study.best_value
#     best_cv_mae = study.best_trial.user_attrs['mae']
#     best_params = study.best_params




#     print("Начало финального обучения на всех данных...")
#     start_time_fit = time.time()
    
#     # Докидываем к лучшим параметрам константы, чтобы собрать финальную модель
#     final_lgb_params = best_params.copy()
#     final_lgb_params.update({
#         'objective': 'mae',
#         'random_state': CONFIG['RANDOM_STATE'],
#         'n_jobs': -1, # Тут можно -1, т.к. мы больше не делаем кросс-валидацию параллельно
#         'verbose': -1,
#         'subsample_freq': 1
#     })
#     final_model = lgb.LGBMRegressor(**final_lgb_params)
#     final_wrapped = TransformedTargetRegressor(
#         regressor=final_model,
#         func=np.log1p,
#         inverse_func=np.expm1
#     )
    
#     final_wrapped.fit(X_train, y_train)
    
#     time_fit = time.time() - start_time_fit
    
#     print("Проверка на X_test...")
#     y_pred_test = final_wrapped.predict(X_test)
    
#     test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
#     test_mae = mean_absolute_error(y_test, y_pred_test)

    
#     mlflow.log_params({f"regressor__{k}": v for k, v in final_lgb_params.items()})
    
#     # Метрики кросс-валидации (какое качество модель ожидала)
#     mlflow.log_metric('cv_best_mape', best_cv_mape)
#     mlflow.log_metric('cv_best_mae', best_cv_mae)
    
#     # МЕТРИКИ НА ТЕСТЕ (РЕАЛЬНОЕ КАЧЕСТВО НА НОВЫХ ДАННЫХ)
#     mlflow.log_metric('test_mape', test_mape)
#     mlflow.log_metric('test_mae', test_mae)
    
#     # Время
#     mlflow.log_metric('time_hpo_sec', time_elapsed)
#     mlflow.log_metric('time_final_fit_sec', time_fit)
    
#     mlflow.sklearn.log_model(final_wrapped, "best_lightgbm_model")

#     print(f"Готово!")
#     print(f"CV MAPE: {best_cv_mape:.4f} | TEST MAPE (Итог): {test_mape:.4f}")
#     print(f"CV MAE:  {best_cv_mae:.0f} | TEST MAE (Итог):  {test_mae:.0f}")


run_name = 'catboost_final'
with mlflow.start_run(run_name=run_name):
    cat_features = X_train.select_dtypes(include=['object', 'category']).columns.to_list()
    print(f'Начало большого поиска Optuna для CatBoost...')
    def objective(trial):
        bootstrap_type = trial.suggest_categorical('bootstrap_type', ['Bayesian', 'Bernoulli', 'MVS'])
                
        cb_params = {
            'allow_writing_files': False,
            'iterations': 5000,
            'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.15, log=True),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 0.1, 20.0, log=True),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'bootstrap_type': bootstrap_type,
            'loss_function': 'MAE',
            'eval_metric': 'MAE',
            'task_type': 'GPU',
            'random_seed': CONFIG['RANDOM_STATE'],
            'thread_count': 1,
            'verbose': 0,
        }

        # Выставляем доп. параметры под выбранный бутстрап
        if bootstrap_type == 'Bayesian':
            cb_params['bagging_temperature'] = trial.suggest_float('bagging_temperature', 0.0, 10.0)
        elif bootstrap_type in ['Bernoulli', 'MVS']:
            cb_params['subsample'] = trial.suggest_float('subsample', 0.4, 1.0)


        
        current_model = CatBoostRegressor(**cb_params)
        current_wrapped = TransformedTargetRegressor(
            regressor=current_model,
            func=np.log1p,
            inverse_func=np.expm1
        )
        
        scores = cross_validate(
            estimator=current_wrapped,
            X=X_train,
            y=y_train,
            scoring=scoring,
            cv=cv,
            params={'cat_features': cat_features},
            n_jobs=1
        )
        
        mape = -scores['test_mape'].mean()
        mae = -scores['test_mae'].mean()
        trial.set_user_attr('mae', mae)
        
        return mape



    start_time = time.time()
    optuna.logging.set_verbosity(optuna.logging.INFO)
    sampler = optuna.samplers.TPESampler(seed=CONFIG['RANDOM_STATE'])
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=150)
    time_elapsed = time.time() - start_time
    print(f"Поиск завершен за {time_elapsed / 60:.2f} минут.")

    best_cv_mape = study.best_value
    best_cv_mae = study.best_trial.user_attrs['mae']
    best_params = study.best_params




    print("Начало финального обучения на всех данных...")
    start_time_fit = time.time()
    final_catboost_params = best_params.copy()
    final_catboost_params.update({
        'loss_function': 'MAE',
        'task_type': 'GPU',
        'allow_writing_files': False,
        'eval_metric': 'MAE',
        'random_seed': CONFIG['RANDOM_STATE'],
        'verbose': 0
    })
    final_model = CatBoostRegressor(**final_catboost_params)
    final_wrapped = TransformedTargetRegressor(
        regressor=final_model,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    final_wrapped.fit(X_train, y_train, cat_features=cat_features)
    
    time_fit = time.time() - start_time_fit
    
    print("Проверка на X_test...")
    y_pred_test = final_wrapped.predict(X_test)
    
    test_mape = mean_absolute_percentage_error(y_test, y_pred_test)
    test_mae = mean_absolute_error(y_test, y_pred_test)

    
    mlflow.log_params({f"regressor__{k}": v for k, v in final_catboost_params.items()})
    
    mlflow.log_metric('cv_best_mape', best_cv_mape)
    mlflow.log_metric('cv_best_mae', best_cv_mae)
    
    mlflow.log_metric('test_mape', test_mape)
    mlflow.log_metric('test_mae', test_mae)
    
    mlflow.log_metric('time_hpo_sec', time_elapsed)
    mlflow.log_metric('time_final_fit_sec', time_fit)
    
    mlflow.sklearn.log_model(
        final_wrapped, 
        artifact_path="best_catboost_model",
        serialization_format="cloudpickle"
    )

    print(f"Готово!")
    print(f"CV MAPE: {best_cv_mape:.4f} | TEST MAPE (Итог): {test_mape:.4f}")
    print(f"CV MAE:  {best_cv_mae:.0f} | TEST MAE (Итог):  {test_mae:.0f}")