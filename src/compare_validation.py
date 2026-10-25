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
RESULTS = PROJECT_ROOT / "results"

with (ROOT / "split_manifest.json").open() as file:
    protocol = json.load(file)

with (ROOT / "manifest.json").open() as file:
    manifest = json.load(file)

section = protocol["splits"]["validation"]
start = section["start_index"]
stop = section["stop_index_exclusive"]

# Özgün veri matrisindeki mutlak satır indeksleri.
endpoints = np.load(
    RESULTS / "cnn" / "validation_endpoints.npy"
)

expected = np.load(
    ROOT / "validation_window_endpoints.npy"
)

if not np.array_equal(endpoints, expected):
    raise ValueError("CNN skorlarının zaman indeksleri uyuşmuyor.")

if not ((endpoints >= start) & (endpoints < stop)).all():
    raise ValueError("Doğrulama dışına taşan indeks var.")

target = np.load(
    ROOT / "target.npy", mmap_mode="r"
)[endpoints].astype(bool)

timestamps = np.load(
    ROOT / "timestamps_ns.npy", mmap_mode="r"
)[endpoints]

eligible = np.load(ROOT / "eligible.npy", mmap_mode="r")

if not eligible[endpoints].all():
    raise ValueError("Uygun olmayan değerlendirme noktası var.")

# ECOD skor dosyası doğrulamanın bütün satırlarını içeriyor.
ecod_scores = np.load(
    RESULTS / "ecod" / "validation_scores.npy",
    mmap_mode="r",
)[endpoints - start]

# CNN skor dosyası yalnızca pencere sonlarını içeriyor.
cnn_scores = np.load(
    RESULTS / "cnn" / "validation_scores.npy",
    mmap_mode="r",
)

if len(cnn_scores) != len(endpoints):
    raise ValueError("CNN skor uzunluğu uyuşmuyor.")

# Özgün olay aralıklarını oku.
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

metric_rows = []
event_rows = []

for name, scores in [
    ("ecod", ecod_scores),
    ("cnn", cnn_scores),
]:
    if not np.isfinite(scores).all():
        raise ValueError(f"{name}: geçersiz skor var.")

    with (RESULTS / name / "run.json").open() as file:
        run = json.load(file)

    threshold = run["threshold"]
    prediction = scores > threshold

    tn, fp, fn, tp = confusion_matrix(
        target, prediction, labels=[False, True]
    ).ravel()

    # Zaman eksenindeki bir boşluk yeni alarm bölümü başlatır.
    continuous = np.r_[
        False,
        np.diff(timestamps) == step.value,
    ]
    previous_alarm = np.r_[False, prediction[:-1]]

    alarm_episodes = int((
        prediction & ~(previous_alarm & continuous)
    ).sum())

    detected_events = 0
    evaluable_events = 0

    for event_id, records in events.groupby("ID"):
        event_mask = np.zeros(len(endpoints), dtype=bool)

        for record in records.itertuples(index=False):
            left = np.searchsorted(
                timestamps,
                record.StartTime.value,
                side="left",
            )
            right = np.searchsorted(
                timestamps,
                (record.EndTime + step).value,
                side="left",
            )
            event_mask[left:right] = True

        hits = event_mask & prediction
        evaluable = bool(event_mask.any())
        detected = bool(hits.any())

        evaluable_events += int(evaluable)
        detected_events += int(detected)

        event_rows.append({
            "model": name,
            "ID": event_id,
            "evaluable_points": int(event_mask.sum()),
            "alarm_points": int(hits.sum()),
            "detected": detected,
        })

    metric_rows.append({
        "model": name,
        "evaluated_points": len(target),
        "threshold": threshold,
        "precision": precision_score(
            target, prediction, zero_division=0
        ),
        "recall": recall_score(
            target, prediction, zero_division=0
        ),
        "point_f0_5": fbeta_score(
            target, prediction, beta=0.5, zero_division=0
        ),
        "average_precision": average_precision_score(
            target, scores
        ),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "alarm_points": int(prediction.sum()),
        "alarm_episodes": alarm_episodes,
        "evaluable_event_ids": evaluable_events,
        "detected_event_ids": detected_events,
    })

comparison = pd.DataFrame(metric_rows)
event_table = pd.DataFrame(event_rows)

comparison.to_csv(
    RESULTS / "validation_comparison.csv", index=False
)
event_table.to_csv(
    RESULTS / "validation_event_comparison.csv", index=False
)

print("Ortak zaman noktalarındaki karşılaştırma:\n")
print(comparison.to_string(index=False))

print("\nOlay başına alarm noktası:\n")
print(
    event_table.pivot(
        index="ID",
        columns="model",
        values="alarm_points",
    ).to_string()
)