import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime, date
from src.shap_utils import get_shap
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
    if key in st.session_state.car_data:
        val = st.session_state.car_data[key]
        
        # Список значений, которые считаются "ИИ ничего не нашел"
        invalid_values = [None, "", "Unknown", "unknown", "null", "nan", "None"]
        
        # Подсвечиваем ЗВЕЗДОЧКОЙ только если ИИ действительно нашел реальное значение
        if val not in invalid_values:
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
                    st.session_state.car_data = {str(k).lower(): v for k, v in parsed.items()}
                    status.update(label="Данные успешно распознаны! Проверьте форму ниже.", state="complete", expanded=False)
                except Exception as e:
                    status.update(label="Произошла ошибка при обращении к ИИ", state="error", expanded=False)
                    st.error(f"Ошибка: {e}")

st.markdown("---")
st.subheader('Шаг 2: Проверка и корректировка параметров')
if st.session_state.llm_fields:
    st.info('Поля, которые заполнил искусственный интеллект, отмечены значком ✨. Проверьте их правильность!')
cd = st.session_state.car_data

col1, col2, col3, col4 = st.columns(4)

with col1:
    brand = st.text_input(get_label("brand", "Марка"), value=cd.get("brand", "lada"))
    car_name = st.text_input(get_label("car_name", "Название машины"), value=cd.get("car_name", "lada granta"))
    
    init_year = int(cd.get("year", 2015)) if cd.get("year") else 2015
    year = st.number_input(get_label("year", "Год выпуска"), max_value=current_year, value=init_year, step=1)
    if year < 1990:
        st.warning('Внимание! Модель плохо предсказывает цену автомобилей с таким годом выпуска. Будьте внимательны и учтите, что предсказанная цена будет менее точна, чем обычно.')
        
    init_mileage = int(cd.get("mileage", 40000)) if cd.get("mileage") else 40000
    mileage = st.number_input(get_label("mileage", "Пробег (км)"), min_value=0, value=init_mileage, step=1000)
    if mileage > 999999:
        st.warning('Внимание! Модель плохо предсказывает цену автомобилей с таким пробегом. Будьте внимательны и учтите, что предсказанная цена будет менее точна, чем обычно.')
        
    eng_types = ['бензин', 'дизель', 'электро', 'газ', 'газ/бензин', 'ГБО']
    parsed_eng = str(cd.get("engine_type", "бензин")).lower()
    eng_idx = eng_types.index(parsed_eng) if parsed_eng in eng_types else 0
    engine_type = st.selectbox(get_label("engine_type", "Тип двигателя"), eng_types, index=eng_idx)

with col2:
    init_power = int(cd.get("power", 106)) if cd.get("power") else 106
    power = st.number_input(get_label("power", "Мощность (л.с.)"), min_value=0, value=init_power, step=1)
    
    init_vol = float(cd.get("engine_vol", 1.6)) if cd.get("engine_vol") else 1.6
    engine_vol = st.number_input(get_label("engine_vol", "Объем двигателя (л)"), min_value=0.0, value=init_vol, step=0.1)
    
    owner_types = ["Частное лицо", "Дилер/Салон"]
    parsed_owner = str(cd.get("owner", "Частное лицо")).lower()
    owner_idx = 1 if ("дилер" in parsed_owner or "салон" in parsed_owner or "фирма" in parsed_owner) else 0
    owner = st.selectbox(get_label("owner", "Владелец"), owner_types, index=owner_idx)

    init_rest = int(cd.get("restyling")) if cd.get("restyling") and str(cd.get("restyling")).isdigit() else None
    restyling = st.number_input(get_label("restyling", "Рестайлинг"), value=init_rest, step=1, min_value=0)
    
    init_gen = int(cd.get("generation")) if cd.get("generation") and str(cd.get("generation")).isdigit() else None
    generation = st.number_input(get_label("generation", "Поколение"), value=init_gen, step=1, min_value=0)

with col3:
    trans_types = ['АКПП', 'МКПП', 'CVT', 'РКПП', 'редуктор']
    parsed_trans = str(cd.get("transmission", "АКПП")).upper()
    if "АВТОМАТ" in parsed_trans: parsed_trans = "АКПП"
    elif "МЕХАНИКА" in parsed_trans: parsed_trans = "МКПП"
    elif "ВАРИАТОР" in parsed_trans: parsed_trans = "CVT"
    elif "РОБОТ" in parsed_trans: parsed_trans = "РКПП"
    trans_idx = trans_types.index(parsed_trans) if parsed_trans in trans_types else 0
    transmission = st.selectbox(get_label("transmission", "Коробка передач"), trans_types, index=trans_idx)
    drive_types = ['передний', 'задний', '4WD', 'двигатель посередине (MID)']
    parsed_drive = str(cd.get("drive_type", "передний")).lower()
    if "полн" in parsed_drive or "4wd" in parsed_drive or "4x4" in parsed_drive:
        drive_idx = 2
    elif "задн" in parsed_drive:
        drive_idx = 1
    elif "сред" in parsed_drive or "mid" in parsed_drive:
        drive_idx = 3
    else:
        drive_idx = 0
    drive_type = st.selectbox(get_label("drive_type", "Привод"), drive_types, index=drive_idx)
    
    equipment = st.text_input(get_label("equipment", "Комплектация"), value=cd.get("equipment", "Unknown"))
    region = st.text_input(get_label("region", "Регион"), value=cd.get("region", "Unknown"))
    special_marks = st.text_input(get_label("special_marks", "Особые отметки (негативные)"), value=cd.get("special_marks", 'нет'))

with col4:
    steering_types = ['левый', 'правый']
    parsed_wheel = str(cd.get("steering_wheel", "левый")).lower()
    wheel_idx = 1 if "прав" in parsed_wheel else 0
    steering_wheel = st.selectbox(get_label("steering_wheel", "Руль"), steering_types, index=wheel_idx)

    parsed_color = str(cd.get("color", "черный")).replace("ё", "е").lower()
    color = st.text_input(get_label("color", "Цвет (вводите через букву \"е\", не \"ё\")"), value=parsed_color)
    
    owners_list = ['1', '2', '3', '4 и более']
    parsed_owners = str(cd.get("owners_count", "1"))
    owners_idx = owners_list.index(parsed_owners) if parsed_owners in owners_list else 0
    owners_count = st.selectbox(get_label("owners_count", "Владельцы"), owners_list, index=owners_idx)

    init_date = date.today()
    if cd.get("ad_date"):
        try:
            init_date = datetime.strptime(str(cd.get("ad_date")), "%Y-%m-%d").date()
        except Exception:
            pass
    ad_date = st.date_input(get_label("ad_date", "Дата размещения объявления"), value=init_date, max_value=date.today(), min_value=date(1999, 1, 1))
    
    parsed_body = str(cd.get("body_type", "Unknown"))
    body_type = st.text_input(get_label("body_type", "Тип кузова (вводите с маленькой буквы)"), value=parsed_body)
st.markdown("---")
st.subheader('Дополнительно (необязательно)')
user_target_price = st.number_input(
    "Введите цену (например, из объявления) в рублях, чтобы проверить её на адекватность:",
    min_value = 0,
    value = 0,
    step = 50000
)



st.markdown("---")
if st.button("Рассчитать цену и влияние факторов", use_container_width=True):
    st.session_state.llm_fields = []
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
        "Особые отметки": [special_marks],
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

    if user_target_price > 0:
        if user_target_price < result['lower_bound']:
            st.error(
                "**Внимание! Цена сильно занижена относительно рыночной!**\n"
                "Рекомендуем тщательно проверить автомобиль на скрытые физические повреждения, "
                "скрученный пробег, а также юридическую чистоту и залоги."
            )
        else:
            st.info("Введенная вами цена находится в пределах рыночной нормы.")

    st.markdown("---")
    st.subheader("Анализ факторов ценообразования (SHAP)")
    with st.spinner('Интерпретация модели...'):
        try:
            shap_data = get_shap(model, df_prepared)
            effects = shap_data["effects_list"]

            top_n = 10
            chart_effects = effects[:top_n]
            rest_effects = effects[top_n:]
            rest_sum = sum(item['effect_rubles'] for item in rest_effects)

            x_labels = ["Базовая (средняя) цена"] + [item['name'] for item in chart_effects]
            y_values = [shap_data["base_rubles"]] + [item["effect_rubles"] for item in chart_effects]
            measures = ["absolute"] + ["relative"] * len(chart_effects)
            if rest_sum != 0:
                x_labels.append("Остальные")
                y_values.append(rest_sum)
                measures.append("relative")
                
            x_labels.append("Итоговая цена")
            y_values.append(shap_data["pred_rubles"])
            measures.append("total")

            fig = go.Figure(go.Waterfall(
                name='Влияние',
                orientation='v',
                measure=measures,
                x=x_labels,
                textposition='outside',
                text=[f"{v/1000:+.0f}k" if m == "relative" else f"{v/1000:.0f}k" for v, m in zip(y_values, measures)],
                y=y_values,
                connector={"line": {"color": "rgb(63, 63, 63)"}},
                decreasing={"marker": {"color": "#FF4B4B"}}, # Красный для минуса
                increasing={"marker": {"color": "#00CC96"}}, # Зеленый для плюса
                totals={"marker": {"color": "#3b82f6"}}      # Синий для итогов
            ))
            fig.update_layout(
                title="Как формировалась цена (каскадная диаграмма)",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            

            def format_effect(item):
                sign = "🟢 +" if item["effect_rubles"] > 0 else "🔴 -"
                val_str = str(item['value']) if item['value'] is not None else 'Unknown'
                return f"{sign}**{item['name']}** ({val_str}) изменил цену на **{item['effect_rubles']:,}** ₽".replace(",", " ")
            for item in top_effects:
                st.write(format_effect(item))
            with st.expander("Остальные признаки, которые меньше всего повлияли на цену"):
                for item in other_effects:
                    st.write(format_effect(item))

        except Exception as e:
            st.warning(f"Не удалось построить график SHAP: {e}")








    with st.expander("Посмотреть сгенерированные признаки (для отладки)"):
        st.dataframe(df_prepared)
