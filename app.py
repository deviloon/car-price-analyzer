import streamlit as st
import pandas as pd
import numpy as np
import time
from datetime import datetime, date
from src.feature_eng import create_features
from src.llm_parser import parse_car_description
from src.predict import predict_price, load_model

current_year = datetime.now().year

st.set_page_config(page_title='Оценка стоимости авто', page_icon="🚗", layout='centered')

@st.cache_resource
def init_model():
    return load_model('models/best_catboost_model.pkl')

try:
    model = init_model()
except FileNotFoundError as e:
    st.error(f'Ошибка: {e}')
    st.stop() # Останавливаем сайт, если модели нет

api_key = st.secrets.get("OPENROUTER_API_KEY", "")
if 'car_data' not in st.session_state:
    st.session_state.car_data = {}
if "llm_fields" not in st.session_state:
    st.session_state.llm_fields = [] # Здесь будем хранить ключи, которые нашел ИИ

# Функция для подсветки названий полей
def get_label(key, base_name):
    if key in st.session_state.llm_fields:
        return f"{base_name} ✨"
    return base_name

st.title('Оценка стоимости автомобиля')
st.write('Введите параметры автомобиля, чтобы узнать его примерную рыночную стоимость.')

with st.expander('Шаг 1: Автозаполнение формы с помощью ИИ', expanded=True):
    user_description = st.text_area(
        'Введите описание автомобиля своими словами:',
        placeholder="Например: Продаю ладу гранту 15 года, 2 владельца, пробег 40к, механика, 106 сил"
    )
    if st.button('Распарсить через ИИ'):
        if not api_key:
            st.error("Ключ OPENROUTER_API_KEY не найден в файле .streamlit/secrets.toml")
        elif not user_description.strip():
            st.warning("Пожалуйста, введите описание автомобиля.")
        else:
            with st.status("Нейросеть извлекает характеристики...", expanded=True) as status:
                st.write("Подключение к OpenRouter...")
                try:
                    time.sleep(0.2) 
                    st.write("Извлечение параметров автомобиля...")
                    parsed = parse_car_description(user_description, api_key)
                    st.write("Заполнение формы...")
                    st.session_state.car_data = parsed
                    status.update(label="Данные успешно распознаны! Проверьте форму ниже.", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Произошла ошибка при обращении к ИИ", state="error", expanded=False)
                    st.error(f"Ошибка: {e}")



col1, col2, col3, col4 = st.columns(4)

with col1:
    brand = st.text_input("Марка", value="lada")
    car_name = st.text_input("Название машины", value="lada granta")
    year = st.number_input("Год выпуска", max_value=current_year, value=2015, step=1)
    if year < 1990:
        st.warning(f'Внимание! Модель плохо предсказывает цену автомобилей с таким годом выпуска. Будьте внимательны и учтите, что предсказанная цена будет менее точна, чем обычно.')
    mileage = st.number_input("Пробег (км)", min_value=0, value=40000, step=1000)
    if mileage > 999999:
        st.warning(f'Внимание! Модель плохо предсказывает ццену автомобилей с таким пробегом. Будьте внимательны и учтите, что предсказанная цена будет менее точна, чем обычно.')
    engine_type = st.selectbox("Тип двигателя", ['бензин', 'дизель', 'электро', 'газ', 'газ/бензин', 'ГБО'])
with col2:
    power = st.number_input("Мощность (л.с.)", min_value=0, value=106, step=1)
    engine_vol = st.number_input("Объем двигателя (л)", min_value=0.0, value=1.6, step=0.1)
    owner = st.selectbox("Владелец", ["Частное лицо", "Дилер/Салон"])
    restyling = st.number_input("Рестайлинг", value=None, step=1, min_value=0)
    generation = st.number_input("Поколение", value=None, step=1, min_value=0)

with col3:
    transmission = st.selectbox("Коробка передач", ['АКПП', 'МКПП', 'CVT', 'РКПП', 'редуктор'])
    drive_type = st.selectbox("Привод", ['передний', 'задний', '4WD', 'двигатель посередине (MID)'])
    equipment = st.text_input("Комплектация", value='Unknown')
    region = st.text_input("Регион", value='Unknown')
    special_marks = st.text_input("Особые отметки (негативные)", value=None)

with col4:
    steering_wheel = st.selectbox("Руль", ['левый', 'правый'])
    color = st.text_input("Цвет (вводите через букву \"е\", не \"ё\")", value="черный")
    owners_count = st.selectbox("Владельцы", ['1', '2', '3', '4 и более'])
    ad_date = st.date_input("Дата размещения объявления", value=date.today(), max_value=date.today(), min_value=date(1999, 1, 1))
    body_type = st.text_input("Тип кузова (вводите с маленькой буквы)", value='Unknown')

st.markdown("---") # Разделительная линия
if st.button("Рассчитать цену", use_container_width=True):
    raw_data = {
        "Марка": [brand],
        "Название машины": [car_name],
        "Год": [year],
        "Пробег": [mileage],
        "Тип двигателя": [engine_type],
        "Мощность": [power],
        "Объем двигателя": [engine_vol],
        "Владелец": [owner],
        "Рестайлинг": [restyling],
        "Поколение": [generation],
        "Коробка передач": [transmission],
        "Привод": [drive_type],
        "Комплектация": [equipment],
        "Регион": [region],
        "Особые отметки": [special_marks.strip() if special_marks and special_marks.strip() else None],
        "Руль": [steering_wheel],
        "Цвет": [color],
        "Владельцы": [owners_count],
        "Дата размещения объявления": [ad_date],
        "Тип кузова": [body_type]
    }
    df_raw = pd.DataFrame(raw_data)
    df_prepared = create_features(df_raw)
    result = predict_price(model, df_prepared, mape=0.1539)

    price_str = f"{result['price']:,}".replace(",", " ") + " ₽"

    delta_val = result['upper_bound'] - result['price']
    delta_str = f"± {delta_val:,}".replace(",", " ") + f" ₽ (Погрешность ~{result['mape_percent']}%)"

    st.success("Расчет успешно завершен!")
    st.metric(label="Рекомендуемая цена", value=price_str, delta=delta_str, delta_color="off")

    with st.expander("Посмотреть сгенерированные признаки (для отладки)"):
        st.dataframe(df_prepared)
