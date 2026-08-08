import joblib
import numpy as np
import os

def load_model(model_path="models/best_catboost_model.pkl"):
    """
    Загружает модель, сохраненную через joblib.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")
    
    return joblib.load(model_path)

def predict_price(model, df_prepared, mape=0.1539):
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
        
    lower_bound = price * (1 - (2*mape))
    upper_bound = price * (1 + mape)
    
    # Округляем до тысяч рублей для красивого вывода
    price_rounded = round(price, -3)
    lower_rounded = round(lower_bound, -3)
    upper_rounded = round(upper_bound, -3)
    
    return {
        "price": int(price_rounded),
        "lower_bound": int(lower_rounded),
        "upper_bound": int(upper_rounded),
        "mape_percent": int(mape * 100)
    }
