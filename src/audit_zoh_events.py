import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "test"

with (PROJECT_ROOT / "config.json").open() as file:
    config = json.load(file)

DATA_ROOT = PROJECT_ROOT / config["data_root"]

with (ROOT / "manifest.json").open() as file:
    manifest = json.load(file)

step = pd.Timedelta(manifest["frequency"])

endpoints = np.load(OUTPUT / "endpoints.npy")
grid = np.load(
    ROOT / "timestamps_ns.npy", mmap_mode="r"
)[endpoints]

labels = pd.read_csv(
    ROOT / "selected_label_intervals.csv"
)

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(labels[column], utc=True)

inventory = pd.read_csv(
    OUTPUT / "event_duration_analysis.csv"
)

labels = labels[
    (labels["Category"] == "Anomaly")
    & labels["ID"].isin(inventory["ID"])
]

rows = []

# Her kanalı yalnızca bir kez oku.
for channel in config["channels"]:
    frame = pd.read_pickle(
        DATA_ROOT / "channels" / f"{channel}.zip"
    )

    frame.index = pd.to_datetime(frame.index, utc=True)
    frame = frame.sort_index(kind="stable")
    frame = frame[
        ~frame.index.duplicated(keep="last")
    ]

    raw_time = frame.index.asi8

    selected = labels[labels["Channel"] == channel]

    for record in selected.itertuples(index=False):
        event_start = record.StartTime.value
        event_end = record.EndTime.value

        # Özgün kapalı olay aralığındaki ham ölçüm sayısı.
        raw_left = np.searchsorted(
            raw_time, event_start, side="left"
        )
        raw_right = np.searchsorted(
            raw_time, event_end, side="right"
        )

        # Etiket kutularının eşleştiği test zamanları.
        left = np.searchsorted(
            grid, event_start, side="left"
        )
        right = np.searchsorted(
            grid, event_end + step.value, side="left"
        )

        event_grid = grid[left:right]

        # Ön işlemede olduğu gibi son geçmiş ölçümü bul.
        source_positions = (
            np.searchsorted(
                raw_time, event_grid, side="right"
            ) - 1
        )

        if (source_positions < 0).any():
            raise ValueError("Geçmiş ölçüm bulunamadı.")

        source_times = raw_time[source_positions]

        source_inside_event = (
            (source_times >= event_start)
            & (source_times <= event_end)
        )

        retained_sources = np.unique(
            source_positions[source_inside_event]
        )

        rows.append({
            "ID": record.ID,
            "channel": channel,
            "start": str(record.StartTime),
            "end": str(record.EndTime),
            "raw_points_inside_interval": raw_right - raw_left,
            "labelled_test_grid_points": len(event_grid),
            "grid_points_using_event_observation": int(
                source_inside_event.sum()
            ),
            "distinct_event_observations_retained": len(
                retained_sources
            ),
            "event_observation_represented": bool(
                source_inside_event.any()
            ),
        })

    print(f"{channel}: kontrol tamamlandı.", flush=True)
    del frame, raw_time

detail = pd.DataFrame(rows)

detail.to_csv(
    OUTPUT / "zoh_event_channel_audit.csv",
    index=False,
)

# Olayın en az bir kanalında olay içi ölçüm temsil edilmiş mi?
summary = detail.groupby("ID", as_index=False).agg(
    channels_with_event_observation=(
        "event_observation_represented", "nunique"
    ),
    any_event_observation_represented=(
        "event_observation_represented", "any"
    ),
    any_raw_observation_inside_interval=(
        "raw_points_inside_interval",
        lambda values: bool((values > 0).any()),
    ),
)

# Kanal sayısını, aynı kanaldaki olası çoklu kayıtları
# iki kez saymadan hesapla.
represented_channels = (
    detail[detail["event_observation_represented"]]
    .groupby("ID")["channel"]
    .nunique()
)

summary["channels_with_event_observation"] = (
    summary["ID"]
    .map(represented_channels)
    .fillna(0)
    .astype(int)
)

summary = inventory.merge(
    summary, on="ID", validate="one_to_one"
)

summary.to_csv(
    OUTPUT / "zoh_event_summary.csv",
    index=False,
)

print("\nOlay bazında ZOH temsili:\n")

print(summary[[
    "ID",
    "union_duration_seconds",
    "channels_with_event_observation",
    "any_raw_observation_inside_interval",
    "any_event_observation_represented",
    "ecod_detected",
    "cnn_detected",
]].to_string(index=False))