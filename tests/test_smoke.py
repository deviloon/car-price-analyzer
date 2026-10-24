import unittest

import pandas as pd

from src.feature_eng import create_features
from src.predict import (
    DEFAULT_MODEL_PATH, load_model, load_interval_models, predict_price,
)


class PredictionSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_MODEL_PATH.exists():
            raise unittest.SkipTest(f"model not found: {DEFAULT_MODEL_PATH}")
        cls.model = load_model(DEFAULT_MODEL_PATH)
        try:
            cls.interval_models = load_interval_models()
        except FileNotFoundError as e:
            raise unittest.SkipTest(f"interval models not found: {e}")

    def test_sample_prediction(self):
        raw_data = {
            "Марка": ["lada"],
            "Название машины": ["Лада Гранта"],
            "Год": [2015],
            "Пробег": [40000],
            "Тип двигателя": ["бензин"],
            "Мощность": [87],
            "Объем двигателя": [1.6],
            "Владелец": ["Частное лицо"],
            "Рестайлинг": [1],
            "Поколение": [1],
            "Коробка передач": ["МКПП"],
            "Привод": ["передний"],
            "Комплектация": ["Unknown"],
            "Регион": ["Unknown"],
            "Особые отметки": ["нет"],
            "Руль": ["левый"],
            "Цвет": ["черный"],
            "Владельцы": ["1"],
            "Дата размещения объявления": ["2026-10-23"],
            "Тип кузова": ["седан"],
        }

        prepared = create_features(pd.DataFrame(raw_data))

        self.assertEqual(prepared['Год размещения'].iloc[0], 2025)
        self.assertEqual(prepared['Поколение'].iloc[0], '1.0')
        self.assertEqual(prepared['Рестайлинг'].iloc[0], '1.0')
        # «Модель» должна попасть в словарь обучения, а не остаться всей строкой
        self.assertEqual(prepared['Модель'].astype(str).iloc[0], 'Гранта')

        result = predict_price(self.model, prepared, self.interval_models)

        self.assertEqual(
            set(result),
            {"price", "lower_bound", "upper_bound", "coverage_percent"},
        )
        self.assertGreater(result["price"], 0)
        self.assertLessEqual(result["lower_bound"], result["price"])   # интервал накрывает цену
        self.assertGreaterEqual(result["upper_bound"], result["price"])
        self.assertEqual(result["coverage_percent"], 90)


if __name__ == "__main__":
    unittest.main()
