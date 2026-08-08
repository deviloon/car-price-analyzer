import shap
import numpy as np
from catboost import Pool
import pandas as pd

def get_shap(model, df):
    """
    Рассчитывает SHAP-значения, переводит их из log1p в рубли
    и готовит данные для интерактивного графика и текстового отчета.
    """
    for col in df.select_dtypes(include=['category']).columns:
        df[col] = df[col].astype(str)

    cb_model = model.regressor_
    if hasattr(cb_model, 'feature_names_') and cb_model.feature_names_:
        df = df[cb_model.feature_names_]
    cat_indices = cb_model.get_cat_feature_indices()
    text_indices = cb_model.get_text_feature_indices()

    pool = Pool(df, cat_features=cat_indices, text_features=text_indices)
    cb_shap = cb_model.get_feature_importance(pool, type='ShapValues')

    log_vals = cb_shap[0, :-1]  # Влияние фичей в логарифмах
    log_base = cb_shap[0, -1]   # Базовая цена в логарифмах
    
    feature_names = df.columns.tolist()
    data_values = df.iloc[0].values

    base_rubles = np.expm1(log_base) # Базовая цена среднего авто
    pred_rubles = np.expm1(log_base + np.sum(log_vals)) # Итоговая цена
    
    diff_rubles = pred_rubles - base_rubles # Разница в рублях
    sum_log_vals = np.sum(log_vals) # Разница в логарифмах

    # Пропорционально распределяем рубли по признакам
    if sum_log_vals != 0:
        rubles_vals = log_vals * (diff_rubles / sum_log_vals)
    else:
        rubles_vals = log_vals * 0

    feature_effects = []
    for i in range(len(feature_names)):
        # Округляем до тысяч для красоты
        effect = round(rubles_vals[i], -3) 
        if effect != 0:
            feature_effects.append({
                "name": feature_names[i],
                "value": data_values[i],
                "effect_rubles": int(effect)
            })

    feature_effects.sort(key=lambda x: abs(x["effect_rubles"]), reverse=True)

    return {
        "base_rubles": int(base_rubles),
        "pred_rubles": int(pred_rubles),
        "effects_list": feature_effects
    }
