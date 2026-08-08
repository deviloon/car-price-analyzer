import pandas as pd
import numpy as np
from datetime import datetime

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

    if 'Дата размещения объявления' in df.columns:
        df['Дата размещения объявления'] = pd.to_datetime(df['Дата размещения объявления'], format='mixed', errors='coerce')
        df['Год размещения'] = df['Дата размещения объявления'].dt.year
        df = df.drop(columns=['Дата размещения объявления'])
    elif 'Год размещения' not in df.columns:
        df['Год размещения'] = datetime.now().year # Берем текущий год, если даты нет

    df['Год размещения'] = df['Год размещения'].fillna(datetime.now().year)

    if 'Год' in df.columns:
        df['Возраст'] = df['Год размещения'] - df['Год']
        if 'Пробег' in df.columns:
            df['Пробег за год'] = df['Пробег'] / df['Возраст'].replace(0, 1)

    for col in ['Рестайлинг', 'Поколение']:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown').astype(str)

    cols_to_drop = ['Возраст', 'Название машины']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    cat_cols = df.select_dtypes(include=['object', 'string']).columns.to_list()
    df[cat_cols] = df[cat_cols].astype('category')

    if 'Комплектация' in df.columns:
        text_features = ['Комплектация']
        df['Комплектация'] = df['Комплектация'].astype(str)

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

    df['Владельцы'] = df['Владельцы'].replace(owners_mapping)
    df['Есть особые отметки'] = df['Особые отметки'].notna().astype(int)

    return df