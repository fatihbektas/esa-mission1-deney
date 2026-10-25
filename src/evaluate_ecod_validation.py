import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "ecod"

with (ROOT / "split_manifest.json").open() as file:
    protocol = json.load(file)

with (OUTPUT / "run.json").open() as file:
    run = json.load(file)

with (ROOT / "manifest.json").open() as file:
    manifest = json.load(file)

section = protocol["splits"]["validation"]
start = section["start_index"]
stop = section["stop_index_exclusive"]

timestamps = np.load(
    ROOT / "timestamps_ns.npy", mmap_mode="r"
)[start:stop]

target = np.load(
    ROOT / "target.npy", mmap_mode="r"
)[start:stop].astype(bool)

eligible = np.load(
    ROOT / "eligible.npy", mmap_mode="r"
)[start:stop]

scores = np.load(
    OUTPUT / "validation_scores.npy", mmap_mode="r"
)

if len(scores) != stop - start:
    raise ValueError("Skor uzunluğu doğrulama bölümüyle uyuşmuyor.")

threshold = run["threshold"]
valid = eligible & np.isfinite(scores)

if not valid.any():
    raise ValueError("Değerlendirilebilir nokta yok.")

prediction = valid & (scores > threshold)

# 1. Nokta bazlı değerlendirme.
y_true = target[valid]
y_pred = prediction[valid]

tn, fp, fn, tp = confusion_matrix(
    y_true, y_pred, labels=[False, True]
).ravel()

point_metrics = {
    "precision": float(
        precision_score(y_true, y_pred, zero_division=0)
    ),
    "recall": float(
        recall_score(y_true, y_pred, zero_division=0)
    ),
    "f0_5": float(
        fbeta_score(y_true, y_pred, beta=0.5, zero_division=0)
    ),
    "average_precision": (
        float(average_precision_score(y_true, scores[valid]))
        if y_true.any() else None
    ),
    "tp": int(tp),
    "fp": int(fp),
    "fn": int(fn),
    "tn": int(tn),
    "evaluated_points": int(valid.sum()),
}

# 2. Özgün olay aralıklarıyla tanısal eşleştirme.
labels = pd.read_csv(ROOT / "selected_label_intervals.csv")

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(labels[column], utc=True)

val_start = pd.Timestamp(protocol["validation_start_utc"])
val_end = pd.Timestamp(protocol["test_start_utc"])
step = pd.Timedelta(manifest["frequency"])

events = labels[
    (labels["Category"] == "Anomaly")
    & (labels["StartTime"] < val_end)
    & (labels["EndTime"] >= val_start)
]

rows = []

for event_id, records in events.groupby("ID"):
    # Kanalların aralıklarını tek geniş aralığa çevirmiyoruz.
    # Her aralığı ayrı eşleyip birleşimini alıyoruz.
    event_mask = np.zeros(len(timestamps), dtype=bool)

    for record in records.itertuples(index=False):
        left = np.searchsorted(
            timestamps, record.StartTime.value, side="left"
        )
        right = np.searchsorted(
            timestamps,
            (record.EndTime + step).value,
            side="left",
        )
        event_mask[left:right] = True

    evaluable = event_mask & valid
    hits = np.flatnonzero(evaluable & prediction)

    first_alarm = (
        str(pd.to_datetime(timestamps[hits[0]], unit="ns", utc=True))
        if len(hits) else None
    )

    rows.append({
        "ID": event_id,
        "start": str(records["StartTime"].min()),
        "end": str(records["EndTime"].max()),
        "evaluable_points": int(evaluable.sum()),
        "alarm_points": len(hits),
        "detected": bool(len(hits)),
        "first_alarm_utc": first_alarm,
    })

event_table = pd.DataFrame(rows).sort_values("start")
event_table.to_csv(
    OUTPUT / "validation_event_hits.csv", index=False
)

# Birbirini izleyen alarm noktalarını alarm bölümü olarak say.
# Dışlanan noktalar alarm bölümünü keser.
alarm_episodes = int(
    (prediction & ~np.r_[False, prediction[:-1]]).sum()
)

report = {
    "threshold": threshold,
    "threshold_changed": False,
    "point_metrics": point_metrics,
    "alarm_episodes": alarm_episodes,
    "event_ids": len(rows),
    "evaluable_event_ids": sum(
        row["evaluable_points"] > 0 for row in rows
    ),
    "detected_event_ids": sum(row["detected"] for row in rows),
    "event_matching": "intersection with (t - 30 seconds, t] bins",
    "official_esa_event_metrics": False,
}

with (OUTPUT / "validation_metrics.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))
print("\nOlay bazında tespit tablosu:\n")
print(event_table.to_string(index=False))