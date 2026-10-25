import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

with (PROJECT_ROOT / "config.json").open() as file:
    config = json.load(file)

data_root = PROJECT_ROOT / config["data_root"]
summary = []

for channel in config["channels"]:
    path = data_root / "channels" / f"{channel}.zip"

    # Yalnızca güvenilir ESA veri dosyalarını okuyun.
    frame = pd.read_pickle(path)

    if channel not in frame.columns:
        raise ValueError(
            f"{channel}: beklenen sütun bulunamadı. "
            f"Mevcut sütunlar: {list(frame.columns)}"
        )

    timestamps = pd.to_datetime(frame.index, utc=True, errors="raise")

    if timestamps.hasnans:
        raise ValueError(f"{channel}: eksik zaman damgası var.")

    values = pd.to_numeric(frame[channel], errors="raise")

    summary.append({
        "kanal": channel,
        "satir": len(frame),
        "baslangic": timestamps.min(),
        "bitis": timestamps.max(),
        "sirali": timestamps.is_monotonic_increasing,
        "tekrarli_zaman": int(timestamps.duplicated().sum()),
        "eksik_deger": int(values.isna().sum()),
    })

report = pd.DataFrame(summary)

print(report.to_string(index=False))

common_start = report["baslangic"].max().ceil(config["frequency"])
common_end = report["bitis"].min().floor(config["frequency"])

if common_start > common_end:
    raise ValueError("Altı kanalın ortak zaman aralığı yok.")

point_count = (
    (common_end - common_start) // pd.Timedelta(config["frequency"])
) + 1

print("\nOrtak 30 saniyelik zaman ekseni:")
print(f"Başlangıç : {common_start}")
print(f"Bitiş     : {common_end}")
print(f"Nokta     : {point_count:,}")

output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)

report.to_csv(output_dir / "data_inventory.csv", index=False)