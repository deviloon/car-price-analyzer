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
import os
from datetime import datetime
import category_encoders as ce
import joblib

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
mlflow.set_experiment('car-price-analyzer')


# Пайплайн предобработки
num_cols = X_train.select_dtypes('number').columns.to_list()
low_card_cols = ['Тип двигателя', 'Коробка передач', 'Привод', 'Руль', 'Цвет', 'Тип кузова']
high_card_cols = ['Марка', 'Регион', 'Модель']
cat_cols = low_card_cols + high_card_cols
X_train[cat_cols] = X_train[cat_cols].astype('category')
X_test[cat_cols] = X_test[cat_cols].astype('category')
preprocessor = ColumnTransformer(
    transformers=[
        ('num', RobustScaler(), num_cols),
        ('low_card', OneHotEncoder(handle_unknown='ignore', sparse_output=False), low_card_cols),
        ('high_card', CatBoostEncoder(handle_unknown='value'), high_card_cols)
    ]
)

pipeline_ridge = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', Ridge())
])

y_train_orig = np.expm1(y_train)

tt_pipeline_ridge = TransformedTargetRegressor(
    regressor=pipeline_ridge,
    func=np.log1p,
    inverse_func=np.expm1
)

scoring_metrics = {
    'mae': 'neg_mean_absolute_error',
    'mape': 'neg_mean_absolute_percentage_error',
    'rmse': 'neg_root_mean_squared_error'
}

run_name = 'baseline_ridge'
with mlflow.start_run(run_name=run_name):
    print('Запуск кросс-валидации...')
    cv_results = cross_validate(
        tt_pipeline_ridge,
        X_train,
        y_train_orig,
        cv=5,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )

    mean_mae = -cv_results['test_mae'].mean()
    mean_mape = -cv_results['test_mape'].mean()
    mean_rmse = -cv_results['test_rmse'].mean()
    
    print("\nМетрики (Baseline):")
    print(f"MAE:  {mean_mae:.2f}")
    print(f"MAPE: {mean_mape:.2f}")
    print(f"RMSE: {mean_rmse:.2f}")

    mlflow.log_metric('cv_mean_mae', mean_mae)
    mlflow.log_metric('cv_mean_mape', mean_mape)
    mlflow.log_metric('cv_mean_rmse', mean_rmse)

    print("Эксперимент успешно сохранен в MLflow")



# 1. Мы пишем свою функцию для создания копии (клона) модели CatBoost
def catboost_sklearn_clone(self):
    klass = self.__class__                 # Выясняем, что это за модель (например, CatBoostRegressor)
    params = self.get_params(deep=False)   # Забираем все её настройки (random_state и т.д.)
    
    # Создаем чистую независимую копию всех настроек
    new_params = {k: copy.deepcopy(v) for k, v in params.items()}
    
    # Создаем точно такую же новую модель с этими настройками
    new_obj = klass(**new_params)
    return new_obj

# 2. А теперь мы "на лету" подменяем стандартный метод клонирования в CatBoost на наш
for cls_name in ["CatBoostRegressor", "CatBoostClassifier", "CatBoostRanker"]:
    if hasattr(catboost, cls_name):
        # Добавляем в класс CatBoost наш метод __sklearn_clone__
        setattr(getattr(catboost, cls_name), "__sklearn_clone__", catboost_sklearn_clone)


        # Наследуемся от оригинального CatBoostRegressor
class PatchedCatBoostRegressor(CatBoostRegressor):
    # Объясняем scikit-learn, как правильно клонировать эту модель
    def __sklearn_clone__(self):
        params = self.get_params(deep=False)
        new_params = {k: copy.deepcopy(v) for k, v in params.items()}
        return PatchedCatBoostRegressor(**new_params)
    
tt_pipeline_catboost_default = TransformedTargetRegressor(
    regressor=PatchedCatBoostRegressor(
    random_state=CONFIG['RANDOM_STATE'],
    cat_features=cat_cols,
    allow_writing_files=False
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
    


run_name = 'catboost_default'
with mlflow.start_run(run_name=run_name):
    print('Запуск кросс-валидации...')
    cv_results = cross_validate(
        tt_pipeline_catboost_default,
        X_train,
        y_train_orig,
        cv=5,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )

    mean_mae = -cv_results['test_mae'].mean()
    mean_mape = -cv_results['test_mape'].mean()
    mean_rmse = -cv_results['test_rmse'].mean()
    
    print("\nМетрики (CatBoostRegressor - default):")
    print(f"MAE:  {mean_mae:.2f}")
    print(f"MAPE: {mean_mape:.2f}")
    print(f"RMSE: {mean_rmse:.2f}")

    mlflow.log_metric('cv_mean_mae', mean_mae)
    mlflow.log_metric('cv_mean_mape', mean_mape)
    mlflow.log_metric('cv_mean_rmse', mean_rmse)





    

    print("Эксперимент успешно сохранен в MLflow")


    
tt_pipeline_xgboost_default = TransformedTargetRegressor(
    regressor=xgb.XGBRegressor(
    random_state=CONFIG['RANDOM_STATE'],
    enable_categorical=True
    ),
    func=np.log1p,
    inverse_func=np.expm1
)

run_name = 'xgboost_default'
with mlflow.start_run(run_name=run_name):
    print('Запуск кросс-валидации...')
    cv_results = cross_validate(
        tt_pipeline_xgboost_default,
        X_train,
        y_train_orig,
        cv=5,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )

    mean_mae = -cv_results['test_mae'].mean()
    mean_mape = -cv_results['test_mape'].mean()
    mean_rmse = -cv_results['test_rmse'].mean()
    
    print("\nМетрики (XGBoostRegressor - default):")
    print(f"MAE:  {mean_mae:.2f}")
    print(f"MAPE: {mean_mape:.2f}")
    print(f"RMSE: {mean_rmse:.2f}")

    mlflow.log_metric('cv_mean_mae', mean_mae)
    mlflow.log_metric('cv_mean_mape', mean_mape)
    mlflow.log_metric('cv_mean_rmse', mean_rmse)

    print("Эксперимент успешно сохранен в MLflow")

tt_pipeline_lightgbm_default = TransformedTargetRegressor(
    regressor=lgb.LGBMRegressor(
    random_state=CONFIG['RANDOM_STATE']
    ),
    func=np.log1p,
    inverse_func=np.expm1
)
    
run_name = 'lightgbm_default'
with mlflow.start_run(run_name=run_name):
    print('Запуск кросс-валидации...')
    cv_results = cross_validate(
        tt_pipeline_lightgbm_default,
        X_train,
        y_train_orig,
        cv=5,
        scoring=scoring_metrics,
        n_jobs=-1,
        return_train_score=False
    )

    mean_mae = -cv_results['test_mae'].mean()
    mean_mape = -cv_results['test_mape'].mean()
    mean_rmse = -cv_results['test_rmse'].mean()
    
    print("\nМетрики (LightGBMRegressor - default):")
    print(f"MAE:  {mean_mae:.2f}")
    print(f"MAPE: {mean_mape:.2f}")
    print(f"RMSE: {mean_rmse:.2f}")

    mlflow.log_metric('cv_mean_mae', mean_mae)
    mlflow.log_metric('cv_mean_mape', mean_mape)
    mlflow.log_metric('cv_mean_rmse', mean_rmse)

    print("Эксперимент успешно сохранен в MLflow")