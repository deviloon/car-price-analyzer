from catboost import CatBoostRegressor
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_prepare(path, target_col='Цена'):
    df = pd.read_parquet(path)
    y = df.pop(target_col).values
    for c in df.select_dtypes(include=['category']).columns:
        df[c] = df[c].astype('object')
    return df, y

X_train_full, y_train_full = load_and_prepare(
    'data/optimized/train_optimized.parquet'
)
X_test, y_test = load_and_prepare('data/optimized/test_optimized.parquet')

cat_features = list(X_train_full.select_dtypes(include=['object']).columns)

X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_train_full, y_train_full, test_size=0.20, random_state=42
)

log_y_tr = np.log(y_tr)
log_y_cal = np.log(y_cal)
log_y_test = np.log(y_test)

print('Обучение нижней квантильной модели (alpha=0.05)')
model_low = CatBoostRegressor(
    loss_function='Quantile:alpha=0.05',
    iterations=1200,
    learning_rate=0.08,
    depth=6,
    cat_features=cat_features,
    random_seed=42,
    verbose=200,
)
model_low.fit(X_tr, log_y_tr)

print('\nОбучение верхней квантильной модели (alpha=0.95)')
model_high = CatBoostRegressor(
    loss_function='Quantile:alpha=0.95',
    iterations=1200,
    learning_rate=0.08,
    depth=6,
    cat_features=cat_features,
    random_seed=42,
    verbose=200,
)
model_high.fit(X_tr, log_y_tr)



print('\nКалибровка CQR')
q_low_cal = model_low.predict(X_cal)
q_high_cal = model_high.predict(X_cal)

# Conformity scores: насколько лог-цена вылезла наружу из интервала
# Если попала внутрь — score будет отрицательным (запас)
scores = np.maximum(q_low_cal - log_y_cal, log_y_cal - q_high_cal)

# Вычисляем квантиль поправки с гарантией покрытия (1 - alpha) = 90%
alpha = 0.10
n = len(scores)
q_level = np.ceil((n + 1) * (1 - alpha)) / n
delta = np.quantile(scores, min(1.0, q_level))

print(f'Калибровочная поправка delta (в лог-шкале): {delta:.4f}')

print('\nОценка на тесте')
q_low_test = model_low.predict(X_test)
q_high_test = model_high.predict(X_test)

lower_bounds = np.exp(q_low_test - delta)
upper_bounds = np.exp(q_high_test + delta)

cov = ((y_test >= lower_bounds) & (y_test <= upper_bounds)).mean()
print(f'-> Честное покрытие на тесте: {cov * 100:.2f}% (цель: 90%)')

widths = upper_bounds - lower_bounds
rel_widths = (upper_bounds - lower_bounds) / y_test

print(f'-> Медианная ширина интервала: {np.median(widths):,.0f} руб.')
print(
    f'-> Медианная ширина относительно цены авто: {np.median(rel_widths) * 100:.1f}%'
)

joblib.dump(
    {'model_low': model_low, 'model_high': model_high, 'delta': delta},
    'models/cqr_models.pkl',
)