import shap
import numpy as np
def get_shap(model, df):
    """
    Рассчитывает SHAP-значения, переводит их из log1p в рубли
    и готовит данные для интерактивного графика и текстового отчета.
    """
    cb_model = model.regressor_
    explainer = shap.TreeExplainer(cb_model)
    shap_values = explainer(df)

    log_base = shap_values.base_values[0]
    log_vals = shap_values.values[0]
    feature_names = shap_values.feature_names
    data_values = shap_values.data[0]

    sum_log_vals = np.sum(log_vals) # Суммарное отклонение в логарифмах
    base_rubles = np.expm1(log_base) # Базовая цена "среднего" авто
    pred_rubles = np.expm1(log_base + sum_log_vals) # Итоговая предсказанная цена
    diff_rubles = pred_rubles - base_rubles # Суммарное отклонение в рублях

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
                "effect_rubles": effect
            })

    feature_effects.sort(key=lambda x: abs(x["effect_rubles"]), reverse=True)

    return {
        "base_rubles": base_rubles,
        "rubles_vals": rubles_vals,
        "data_values": data_values,
        "feature_names": feature_names,
        "effects_list": feature_effects
    }
