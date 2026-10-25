import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

with (PROJECT_ROOT / "config.json").open() as file:
    config = json.load(file)

DATA_ROOT = PROJECT_ROOT / config["data_root"]
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
CHANNELS = config["channels"]

timestamps = np.load(
    PROCESSED_ROOT / "timestamps_ns.npy",
    mmap_mode="r",
)
grid = pd.to_datetime(timestamps, unit="ns", utc=True)

labels = pd.read_csv(DATA_ROOT / "labels.csv")
types = pd.read_csv(DATA_ROOT / "anomaly_types.csv")

labels = labels.merge(
    types[["ID", "Category"]],
    on="ID",
    how="left",
    validate="many_to_one",
)

labels = labels[labels["Channel"].isin(CHANNELS)].copy()

if labels["Category"].isna().any():
    raise ValueError("Kategori bilgisi bulunamayan etiket var.")

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(
        labels[column],
        utc=True,
        errors="raise",
    )

if labels[["StartTime", "EndTime"]].isna().any().any():
    raise ValueError("Eksik etiket zamanı var.")

if (labels["StartTime"] > labels["EndTime"]).any():
    raise ValueError("Başlangıcı bitişinden sonra olan etiket var.")

# Ayrı maskeler, çakışan kategorilerin birbirini ezmesini önler.
categories = {
    "Anomaly": "anomaly",
    "Rare Event": "rare_event",
    "Communication Gap": "communication_gap",
}

unknown = set(labels["Category"]) - set(categories)

if unknown:
    raise ValueError(f"Bilinmeyen kategoriler: {unknown}")

summary = []

for category, filename in categories.items():
    mask = np.lib.format.open_memmap(
        PROCESSED_ROOT / f"{filename}_mask.npy",
        mode="w+",
        dtype="bool",
        shape=(len(grid), len(CHANNELS)),
    )
    mask[:] = False

    for column, channel in enumerate(CHANNELS):
        selected = labels[
            (labels["Channel"] == channel)
            & (labels["Category"] == category)
        ]

        no_grid_point = 0

        for row in selected.itertuples(index=False):
            # Kapalı aralık: StartTime <= zaman <= EndTime
            left = grid.searchsorted(row.StartTime, side="left")
            right = grid.searchsorted(row.EndTime, side="right")

            if left == right:
                no_grid_point += 1

            mask[left:right, column] = True

        summary.append({
            "channel": channel,
            "category": category,
            "annotation_rows": len(selected),
            "positive_grid_points": int(mask[:, column].sum()),
            "annotation_rows_without_grid_point": no_grid_point,
        })

    mask.flush()
    del mask

report = pd.DataFrame(summary)
report.to_csv(
    PROCESSED_ROOT / "label_summary.csv",
    index=False,
)

# Özgün aralıkları olay değerlendirmesi için ayrıca koru.
labels.to_csv(
    PROCESSED_ROOT / "selected_label_intervals.csv",
    index=False,
)

print(report.to_string(index=False))
print("\nEtiket maskeleri kaydedildi.")
