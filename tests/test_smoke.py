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
            "Название машины": ["lada granta"],
            "Год": [2015],
            "Пробег": [40000],
            "Тип двигателя": ["бензин"],
            "Мощность": [106],
            "Объем двигателя": [1.6],
            "Владелец": ["Частное лицо"],
            "Рестайлинг": [0],
            "Поколение": [0],
            "Коробка передач": ["АКПП"],
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

        # Якорь 2025: модель обучена на объявлениях 2025 года, дата из формы
        # (в примере выше — 2026 год) не должна попадать в признаки.
        self.assertEqual(prepared['Год размещения'].iloc[0], 2025)
        # Формат категорий как в обучении: число с одним знаком после точки.
        self.assertEqual(prepared['Поколение'].iloc[0], '0.0')
        self.assertEqual(prepared['Рестайлинг'].iloc[0], '0.0')

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
