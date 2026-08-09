import requests
import json
import re

def extract_json_from_text(text):
    """Вытаскивает чистый JSON из текста ответа нейросети."""
    if not text or not isinstance(text, str):
        return {}
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return {}

def parse_car_description(description: str, api_key: str) -> dict:
    url = "https://openrouter.ai/api/v1/chat/completions"
    
    # OpenRouter просит передавать эти заголовки для бесплатных моделей
    headers = {
        "Authorization": f"Bearer {api_key.strip()}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8501",
        "X-Title": "Car Price Predictor"
    }

    system_prompt = """
Твоя задача — извлечь параметры автомобиля из текста пользователя и вернуть ИХ СТРОГО В ФОРМАТЕ JSON.
JSON должен содержать ВСЕ перечисленные ниже ключи. Никакого дополнительного текста, кроме JSON, возвращать нельзя!

ПРАВИЛА ДЛЯ ОТСУТСТВУЮЩИХ ПАРАМЕТРОВ:
1. Если текстовый/категориальный параметр в тексте не указан, установи для него значение "Unknown".
2. Если числовой параметр в тексте не указан, установи для него значение null.

Разрешенные ключи для JSON и правила их заполнения:
- "brand" (строка, марка авто, например: "lada", "bmw", "toyota")
- "car_name" (строка, название модели, например: "granta", "x5", "camry")
- "year" (целое число, год выпуска, например: 2018)
- "mileage" (целое число, пробег в км, например: 40000)
- "engine_type" (строка, строго одно из: "бензин", "дизель", "электро", "газ", "газ/бензин", "ГБО")
- "power" (целое число, мощность в л.с., например: 150)
- "engine_vol" (число с точкой, объем двигателя в литрах, например: 1.6)
- "owner" (строка, статус продавца/владельца, например: "собственник", "дилер", "физлицо")
- "restyling" (строка, наличие или поколение рестайлинга, например: "рестайлинг", "дорестайлинг")
- "generation" (строка, поколение модели, например: "1 поколение", "XV40", "E60")
- "transmission" (строка, строго одно из: "АКПП", "МКПП", "CVT", "РКПП", "редуктор")
- "drive_type" (строка, тип привода, например: "передний", "задний", "полный")
- "equipment" (строка, название комплектации, например: "Luxe", "Comfort", "Executive")
- "region" (строка, регион или город, например: "Москва", "Краснодарский край")
- "special_marks" (строка, описание повреждений, проблем. эта строка ТОЛЬКО про негативные фактора, а не просто отметки. если в тексте явно нету ничего отрицатльного - оставь Unknown)
- "steering_wheel" (строка, расположение руля: "левый" или "правый")
- "color" (строка, цвет кузова, например: "черный", "белый", "серебристый")
- "owners_count" (строка, строго одно из: "1", "2", "3", "4 и более")
- "ad_date" (строка, дата публикации или упоминаемая дата)
- "body_type" (строка, тип кузова, например: "седан", "внедорожник", "хэтчбек", "универсал", "купе")
    """

    data = {
        "model": "poolside/laguna-s-2.1:free",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"ОПИСАНИЕ АВТО: {description}"}
        ]
    }

    response = requests.post(url, headers=headers, json=data)
    res_json = response.json()
    if 'choices' not in res_json or not res_json['choices']:
        print(f"Ошибка API OpenRouter: {res_json}")
        return {}

    result_text = res_json['choices'][0]['message']['content']
    return extract_json_from_text(result_text)

if __name__ == "__main__":
    TEST_API_KEY = "" 
    
    test_text = "«Tesla Model 3, 2020 год, электро, 351 л.с., редуктор, полный привод, кузов седан. Пробег 52000 км, цвет красный, руль левый. По ПТС 3 владельца. Продает автосалон (дилер). Из особых отметок: была замена лобового стекла и крашено крыло после ДТП. Екатеринбург.»"
    print(f"Тестируем текст: '{test_text}'")
    print("Отправка запроса...")
    
    try:
        parsed_data = parse_car_description(test_text, TEST_API_KEY)
        print("\nУспех! Нейросеть распознала:")
        print(json.dumps(parsed_data, indent=4, ensure_ascii=False))
    except Exception as e:
        print(f"\nОшибка: {e}")