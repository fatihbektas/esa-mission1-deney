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

labels = pd.read_csv(
    ROOT / "selected_label_intervals.csv"
)

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(labels[column], utc=True)

endpoints = np.load(OUTPUT / "endpoints.npy")

timestamps = np.load(
    ROOT / "timestamps_ns.npy", mmap_mode="r"
)[endpoints]

comparison = pd.read_csv(
    OUTPUT / "event_comparison.csv"
)

# Hesaplanmış test değerlendirmesindeki ID'lerle sınırlı tut.
event_ids = comparison["ID"].unique()

labels = labels[
    (labels["Category"] == "Anomaly")
    & labels["ID"].isin(event_ids)
].copy()


def union_duration_seconds(records):
    """Kanallar arasındaki örtüşmeleri iki kez saymadan süreyi bul."""
    intervals = sorted(
        zip(records["StartTime"], records["EndTime"])
    )

    total = 0.0
    left, right = intervals[0]

    for next_left, next_right in intervals[1:]:
        if next_left <= right:
            right = max(right, next_right)
        else:
            total += (right - left).total_seconds()
            left, right = next_left, next_right

    return total + (right - left).total_seconds()


def duration_group(seconds):
    if seconds == 0:
        return "0 sn"
    if seconds < 30:
        return "0–30 sn"
    if seconds < 300:
        return "30 sn–5 dk"
    if seconds < 3600:
        return "5 dk–1 saat"
    return "1 saat ve üzeri"


rows = []

for event_id, records in labels.groupby("ID"):
    duration = union_duration_seconds(records)

    # Mevcut etiket politikasına göre olayın temsil edildiği
    # ortak değerlendirme noktalarının birleşimi.
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

    record = {
        "ID": event_id,
        "start_utc": str(records["StartTime"].min()),
        "end_utc": str(records["EndTime"].max()),
        "channel_count": records["Channel"].nunique(),
        "union_duration_seconds": duration,
        "duration_group": duration_group(duration),
        "grid_points": int(event_mask.sum()),
    }

    for model in ["ecod", "cnn"]:
        selected = comparison[
            (comparison["ID"] == event_id)
            & (comparison["model"] == model)
        ]

        if len(selected) != 1:
            raise ValueError(
                f"{event_id}/{model}: tek değerlendirme satırı bekleniyor."
            )

        result = selected.iloc[0]
        alarm_points = int(result["alarm_points"])

        if int(result["evaluable_points"]) != record["grid_points"]:
            raise ValueError(
                f"{event_id}/{model}: zaman eşlemesi uyuşmuyor."
            )

        record[f"{model}_alarm_points"] = alarm_points
        record[f"{model}_detected"] = alarm_points > 0
        record[f"{model}_point_coverage"] = (
            alarm_points / record["grid_points"]
            if record["grid_points"] else np.nan
        )

    rows.append(record)

inventory = pd.DataFrame(rows).sort_values(
    "union_duration_seconds", ascending=False
)

inventory.to_csv(
    OUTPUT / "event_duration_analysis.csv",
    index=False,
)

# Değerlendirilemeyen olayları yakalama oranının paydasına katma.
evaluable = inventory[inventory["grid_points"] > 0]

order = [
    "0 sn",
    "0–30 sn",
    "30 sn–5 dk",
    "5 dk–1 saat",
    "1 saat ve üzeri",
]

summary_rows = []

for group in order:
    selected = evaluable[
        evaluable["duration_group"] == group
    ]

    if selected.empty:
        continue

    summary_rows.append({
        "duration_group": group,
        "event_ids": len(selected),
        "ecod_detected": int(selected["ecod_detected"].sum()),
        "cnn_detected": int(selected["cnn_detected"].sum()),
        "ecod_hit_fraction": selected["ecod_detected"].mean(),
        "cnn_hit_fraction": selected["cnn_detected"].mean(),
    })

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUTPUT / "duration_group_summary.csv",
    index=False,
)

print("Olay süreleri ve tespitler:\n")

columns = [
    "ID",
    "union_duration_seconds",
    "grid_points",
    "ecod_alarm_points",
    "cnn_alarm_points",
    "ecod_detected",
    "cnn_detected",
]

print(inventory[columns].to_string(index=False))

print("\nSüre grubuna göre tespit:\n")
print(summary.to_string(index=False))

print(
    "\nDeğerlendirilemeyen olay ID'si:",
    int((inventory["grid_points"] == 0).sum()),
)