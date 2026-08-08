import streamlit as st
import pandas as st
import numpy as np
import datetime
from src.feature_eng import create_features
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

st.title('Оценка стоимости автомобиля')
st.write('Введите параметры автомобиля, чтобы узнать его примерную рыночную стоимость.')

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
    seller_type = st.selectbox("Тип продавца", ["Частное лицо", "Дилер/Салон"])
    restyling = st.number_input("Рестайлинг", value=np.nan, step=0)
    generation = st.number_input("Поколение", value=np.nan, step=0)

with col3:
    transmission = st.selectbox("Коробка передач", ['АКПП', 'МКПП', 'CVT', 'РКПП', 'редуктор'])
    drive_type = st.selectbox("Привод", ['передний', 'задний', '4WD', 'двигатель посередине (MID)'])
    equipment = st.text_input("Комплектация", value='Unknown')
    region = st.text_input("Регион", value='Unknown')
    special_marks = st.textInput("Особые отметки (негативные)", value=np.nan)

with col4:
    steering_wheel = st.selectbox("Руль", ['правый', 'левый'])
    color = st.text_input("Цвет (вводите через букву \"е\", не \"ё\")", value="черный")
    owners_count = st.selectbox("Владельцы", ['1', '2', '3', '4 и более'])
    ad_date = st.date_input("Дата размещения объявления", value=datetime.date.today(), max_value=datetime.date.today(), min_value=datetime.date(1999, 1, 1))
    body_type = st.text_input("Тип кузова (вводите с маленькой буквы)", value='Unknown')

st.markdown("---") # Разделительная линия