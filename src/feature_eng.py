import pandas as pd
import numpy as np
import re

def get_brand_tier(brand):
    brand = str(brand).lower().strip()
    
    luxury = {
        "rolls-royce", "bentley", "lamborghini", "ferrari", 
        "aston_martin", "aston martin", "maserati", "bugatti", "mclaren"
    }

    premium = {
        "bmw", "mercedes-benz", "audi", "lexus", "porsche", 
        "land_rover", "land rover", "jaguar", "genesis", "infiniti", 
        "cadillac", "volvo", "jeep", "hongqi", "li"
    }

    economy = {"lada", "daewoo", "zaz", "gaz", "uaz", "vaz"}

    if brand in luxury:
        return "Люкс"
    elif brand in premium:
        return "Премиум"
    elif brand in economy:
        return "Эконом"
    else:
        return "Масс-маркет"

def clean_seller_type(text):
    text = str(text).lower()
    if 'фирма' in text or 'дилер' in text or 'компания' in text:
        return 'Дилер/Салон'
    elif 'частное лицо' in text or 'частник' in text:
        return 'Частное лицо'
    else:
        return 'Unknown'

def extract_model(row):
    full_name = str(row.get('Название машины', ''))
    # Нормализуем исходную марку (например, "aston_martin" -> "aston martin")
    brand_raw = str(row.get('Марка', '')).lower().replace('_', ' ')
    full_name_lower = full_name.lower()
    
    brand_synonyms = {
        'lada': ['лада'],
        'uaz': ['уаз'],
        'gaz': ['газ'],
        'moskvich': ['москвич'],
    }
    
    possible_names = brand_synonyms.get(brand_raw, [brand_raw])
    
    for name in possible_names:
        if full_name_lower.startswith(name):
            return full_name[len(name):].strip()
            
    return full_name


def special_marks(df):
    df['Особые отметки'] = df['Особые отметки'].fillna('нет').str.lower()
    def clean_text(s):
        s = str(s).lower()
        s = re.sub(r'&#\d+;', '', s) # удаляем HTML entities
        s = re.sub(r'[^\w\s]', ' ', s) # удаляем пунктуацию

        val_str = s.strip()
        if val_str in ['none', 'nan', 'null', 'unknown', '']:
            return 'нет'

        return ' '.join(val_str.split())

    df['Особые отметки'] = df['Особые отметки'].apply(clean_text)
    return df


def _format_generation_value(v):
    # Категории обучения записаны как числа с одним знаком после точки
    # ('1.0'...'12.0') плюс 'Unknown', поэтому int из формы нужно приводить к ним.
    if pd.isna(v) or str(v).strip() == '':
        return 'Unknown'
    try:
        return f'{float(v):.1f}'
    except (TypeError, ValueError):
        return str(v)


def create_features(df_input):
    df = df_input.copy()
    if 'Метка' in df.columns:
        df = df.rename(columns={'Метка': 'Марка'})
        
    df['Название машины'] = df.get('Название машины', '').astype(str)
    df['Марка'] = df.get('Марка', '').astype(str)
    df['Модель'] = df.apply(extract_model, axis=1)
    df['Класс бренда'] = df['Марка'].apply(get_brand_tier)

    if 'Мощность' in df.columns and 'Объем двигателя' in df.columns:
        df['Литровая мощность'] = (df['Мощность'] / df['Объем двигателя'].replace(0, np.nan)).fillna(0)

    if 'Владелец' in df.columns:
        df['Тип продавца'] = df['Владелец'].apply(clean_seller_type)
        df = df.drop(columns=['Владелец'])
    else:
        df['Тип продавца'] = 'Unknown'

    # Модель обучена только на объявлениях 2025 года, поэтому в признаки всегда
    # уходит якорный год. Дата из формы остаётся для UX, но модель её не видит:
    # сдвиг рынка во времени закрывает рыночный коэффициент, а не признаки.
    df['Год размещения'] = 2025
    if 'Дата размещения объявления' in df.columns:
        df = df.drop(columns=['Дата размещения объявления'])

    df = special_marks(df)

    if 'Год' in df.columns:
        df['Возраст'] = df['Год размещения'] - df['Год']
        if 'Пробег' in df.columns:
            df['Пробег за год'] = df['Пробег'] / df['Возраст'].replace(0, 1)

    for col in ['Рестайлинг', 'Поколение']:
        if col in df.columns:
            df[col] = df[col].apply(_format_generation_value)

    cols_to_drop = ['Возраст', 'Название машины']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    cat_cols = df.select_dtypes(include=['object', 'string', 'category']).columns.to_list()
    for col in cat_cols:
        df[col] = df[col].astype('category')
    if 'Комплектация' in df.columns:
        df['Комплектация'] = df['Комплектация'].astype(object).fillna('').astype(str)

    owners_mapping = {
    '4 и более': 4,
    '1.0': 1,
    '2.0': 2,
    '3.0': 3,
    1.0: 1,
    2.0: 2,
    3.0: 3,
    '1': 1,
    '2': 2,
    '3': 3
    }

    df['Владельцы'] = (
        df['Владельцы'].astype(object)
        .replace(owners_mapping)
        .infer_objects(copy=False)
    )

    return df
