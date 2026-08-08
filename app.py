import streamlit as st
import pandas as st
from src.feature_eng import create_features
from src.predict import predict_price, load_model

st.set_page_config(page_title='Оценка стоимости авто', page_icon="🚗", layout='centered')

@st.cache_resource
def init_model():
    return load_model('models/best_catboost_model.pkl')

try:
    model = init_model()
except FileNotFoundError as e:
    st.error(f'Ошибка: {e}')
    st.stop() # Останавливаем сайт, если модели нет
    