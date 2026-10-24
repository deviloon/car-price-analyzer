"""Собирает models/category_vocab.json — канонический словарь категорий обучения.

Приложение использует его в src/feature_eng.py, чтобы ввод пользователя в любом
регистре и формате («лада гранта», «Toyota», «москва») приводился к той самой
категории обучающих данных («Гранта», «toyota», «Москва»). CatBoost чувствителен
к регистру, незнакомая категория ломает и цену, и интервал.

Запуск из корня проекта: py scripts/build_category_vocab.py
"""
import json
import re
from pathlib import Path

import pandas as pd


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().replace("ё", "е"))


COLS = ["Марка", "Модель", "Регион", "Цвет", "Тип кузова"]

df = pd.read_parquet("data/optimized/train_optimized.parquet", columns=COLS)
vocab = {}
for col in COLS:
    # при коллизии нормализованных форм побеждает самое частое значение
    mapping = {}
    for value, _count in df[col].astype(str).value_counts().items():
        key = norm(value)
        if key and key not in mapping:
            mapping[key] = value
    vocab[col] = mapping

out = Path("models/category_vocab.json")
out.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")
print(f"Сохранено {out}: " + ", ".join(f"{c}={len(vocab[c])}" for c in COLS))
