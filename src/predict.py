import joblib
import numpy as np
import os
from pathlib import Path

MODEL_MAPE = 0.1539

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best_catboost_model.pkl"
CQR_BUNDLE_PATH = BASE_DIR / "models" / "cqr_models.pkl"
INTERVAL_COVERAGE_PERCENT = 90 

def load_model(model_path=DEFAULT_MODEL_PATH):
    """
    Загружает модель, сохраненную через joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")
    
    return joblib.load(model_path)

def load_interval_models():
    """
    CQR-интервалы: две квантильные CatBoost (alpha=0.05/0.95) на log(цены),
    обучены в models/04_interval.py. delta — конформальная поправка с
    калибровочной выборки.
    """
    if not CQR_BUNDLE_PATH.exists():
        raise FileNotFoundError(
            f"Не найден {CQR_BUNDLE_PATH}. Пересоздается запуском models/04_interval.py."
        )
    return joblib.load(CQR_BUNDLE_PATH)

def predict_price(model, df_prepared, interval_models):
    """
    Делает предсказание и рассчитывает диапазон цен.
    """
    cat_model = model.regressor_
    expected_features = cat_model.feature_names_
    missing_cols = [col for col in expected_features if col not in df_prepared.columns]
    if missing_cols:
        raise ValueError(f"В подготовленных данных не хватает колонок: {missing_cols}")
    df_final = df_prepared[expected_features].copy()
    cat_cols = df_final.select_dtypes(include=['category']).columns
    if len(cat_cols) > 0:
        df_final[cat_cols] = df_final[cat_cols].astype('object')

    price_pred = model.predict(df_final)
    
    # Если возвращается массив, берем первое значение
    if isinstance(price_pred, np.ndarray):
        price = price_pred[0]
    else:
        price = price_pred
        
    cqr = interval_models  # dict: model_low, model_high, delta
    # Квантильные модели обучены на log(цены): их выход — логарифм,
    # поэтому возвращаемся в рубли экспонентой (как в models/04_interval.py)
    q_low = float(cqr["model_low"].predict(df_final)[0])
    q_high = float(cqr["model_high"].predict(df_final)[0])
    lower = np.exp(q_low - cqr["delta"])
    upper = np.exp(q_high + cqr["delta"])

    price_rounded = round(price, -3)
    # нижнюю границу округляем вниз, верхнюю — вверх: округление не должно
    # сужать интервал
    lower_rounded = int(np.floor(lower / 1000) * 1000)
    upper_rounded = int(np.ceil(upper / 1000) * 1000)

    return {
        "price": int(price_rounded),
        "lower_bound": lower_rounded,
        "upper_bound": upper_rounded,
        "coverage_percent": INTERVAL_COVERAGE_PERCENT,
    }
