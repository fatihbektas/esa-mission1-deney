import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "test"

with (ROOT / "manifest.json").open() as file:
    manifest = json.load(file)

step = pd.Timedelta(manifest["frequency"])

endpoints = np.load(OUTPUT / "endpoints.npy")
timestamps = np.load(
    ROOT / "timestamps_ns.npy", mmap_mode="r"
)[endpoints]

labels = pd.read_csv(
    ROOT / "selected_label_intervals.csv"
)

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(labels[column], utc=True)

inventory = pd.read_csv(
    OUTPUT / "zoh_event_summary.csv"
)

labels = labels[
    (labels["Category"] == "Anomaly")
    & labels["ID"].isin(inventory["ID"])
]

rows = []

for model in ["ecod", "cnn"]:
    scores = np.load(
        OUTPUT / f"{model}_scores.npy",
        mmap_mode="r",
    )

    with (
        PROJECT_ROOT / "results" / model / "run.json"
    ).open() as file:
        run = json.load(file)

    threshold = run["threshold"]

    if len(scores) != len(timestamps):
        raise ValueError(f"{model}: skor uzunluğu uyuşmuyor.")

    for event_id, records in labels.groupby("ID"):
        event_mask = np.zeros(len(timestamps), dtype=bool)

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

        values = scores[event_mask]

        if len(values) == 0:
            raise ValueError(f"{event_id}: değerlendirme noktası yok.")

        maximum = float(values.max())

        rows.append({
            "model": model,
            "ID": event_id,
            "points": len(values),
            "score_min": float(values.min()),
            "score_median": float(np.median(values)),
            "score_max": maximum,
            "threshold": threshold,
            "max_minus_threshold": maximum - threshold,
            "alarm_points": int((values > threshold).sum()),
        })

detail = pd.DataFrame(rows)

detail = detail.merge(
    inventory[[
        "ID",
        "union_duration_seconds",
        "channels_with_event_observation",
    ]],
    on="ID",
    validate="many_to_one",
)

detail.to_csv(
    OUTPUT / "event_score_analysis.csv",
    index=False,
)

for model in ["ecod", "cnn"]:
    table = detail[detail["model"] == model].sort_values(
        "max_minus_threshold",
        ascending=False,
    )

    print(f"\n{model.upper()} — olay skorları:\n")

    print(table[[
        "ID",
        "union_duration_seconds",
        "score_median",
        "score_max",
        "threshold",
        "max_minus_threshold",
        "alarm_points",
    ]].to_string(index=False))