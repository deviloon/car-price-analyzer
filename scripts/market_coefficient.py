import argparse
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.feature_eng import create_features
from src.predict import DEFAULT_MODEL_PATH, load_model

ANCHOR_DATE = "2025-06-15"
K_MIN, K_MAX = 0.85, 1.20
BODY_MAP = {
    "suv": "джип/suv 5 дв.",
    "хэтчбек": "хэтчбек 5 дв.",
    "внедорожник": "джип/suv 5 дв.",
    "": "Unknown",
}

def fmt_gen(v):
    if v in ("", None):
        return "Unknown"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return "Unknown"

def latest_listings(explicit=None):
    if explicit:
        return Path(explicit)
    files = sorted(Path("data/market").glob("listings_*.json"))
    if not files:
        raise SystemExit("Нет data/market/listings_*.json — положи свежий парсинг или укажи путь аргументом")
    return files[-1]

def build_rows(listings):
    rows = []
    for L in listings:
        if L.get("mileage_unknown") or L.get("power_outlier") or L.get("engine_outlier"):
            continue
        if not L.get("engine_type") or L.get("price_is_from"):
            continue
        if L.get("special_marks") not in ("нет", "", None):
            continue
        owners = L.get("owners")
        rows.append({
            "Марка": L.get("brand_slug", ""),
            "Название машины": L.get("name", ""),
            "Год": int(L["year"]),
            "Пробег": int(L.get("mileage") or 0),
            "Тип двигателя": L.get("engine_type"),
            "Мощность": int(L.get("power") or 0),
            "Объем двигателя": float(L.get("engine_volume") or 0),
            "Владелец": "Дилер/Салон" if L.get("seller_type") == "dealer" else "Частное лицо",
            "Рестайлинг": fmt_gen(L.get("restyling")),
            "Поколение": fmt_gen(L.get("generation")),
            "Коробка передач": L.get("transmission", ""),
            "Привод": L.get("drive", ""),
            "Комплектация": L.get("complectation") or "Unknown",
            "Регион": L.get("region") or "Unknown",
            "Особые отметки": "нет",
            "Руль": L.get("wheel") or "Unknown",
            "Цвет": L.get("color") or "Unknown",
            "Владельцы": owners if owners not in ("", None) else np.nan,
            "Тип кузова": BODY_MAP.get(L.get("body_type", ""), L.get("body_type", "")),
            "Дата размещения объявления": ANCHOR_DATE,
            "_price": float(L["price"]),
        })
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="путь к listings json (по умолчанию последний в data/market)")
    args = parser.parse_args()

    src = latest_listings(args.path)
    listings = json.loads(src.read_text(encoding="utf-8"))
    rows = build_rows(listings)
    if len(rows) < 500:
        raise SystemExit(f"После фильтров осталось {len(rows)} объявлений — слишком мало, проверь данные")

    df = pd.DataFrame(rows)
    prices = df.pop("_price").values

    model = load_model(DEFAULT_MODEL_PATH)
    cb = model.regressor_
    prepared = create_features(df.copy())
    prepared = prepared[cb.feature_names_].copy()
    for c in prepared.select_dtypes(include=["category"]).columns:
        prepared[c] = prepared[c].astype("object")
    pred = np.asarray(model.predict(prepared))

    log_ratio = np.log(prices / pred)
    k = float(np.exp(np.median(log_ratio)))
    rng = np.random.default_rng(42)
    boots = np.array([np.exp(np.median(rng.choice(log_ratio, size=len(log_ratio)))) for _ in range(3000)])
    ci = [float(x) for x in np.percentile(boots, [2.5, 97.5])]

    clamped = not (K_MIN <= k <= K_MAX)
    if clamped:
        print(f"ВНИМАНИЕ: k={k:.3f} вне клампа [{K_MIN}, {K_MAX}] — вероятно, пора переобучать модель. Применяю кламп.")
        k = float(np.clip(k, K_MIN, K_MAX))

    payload = {
        "k": round(k, 4),
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "source": "drom parser",
        "n_listings": len(rows),
        "n_total": len(listings),
        "source_file": src.name,
        "anchor_year": 2025,
        "bootstrap_ci": [round(c, 4) for c in ci],
        "clamped": clamped,
    }
    Path("models/market_coefficient.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    with open("data/market_history.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    print(f"k = {payload['k']}  (95% ДИ {ci[0]:.3f}–{ci[1]:.3f}, n={len(rows)})")
    print(f"Сохранено: models/market_coefficient.json, история дополнена.")

if __name__ == "__main__":
    main()