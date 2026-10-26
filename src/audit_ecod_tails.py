from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "test"

features = np.load(ROOT / "features.npy", mmap_mode="r")
endpoints = np.load(OUTPUT / "endpoints.npy")
timestamps = np.load(
    ROOT / "timestamps_ns.npy", mmap_mode="r"
)[endpoints]

labels = pd.read_csv(
    ROOT / "selected_label_intervals.csv"
)

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(labels[column], utc=True)

with np.load(
    PROJECT_ROOT / "results" / "ecod" / "model.npz"
) as model:
    reference = model["reference"].copy()
    minimum = model["minimum"].copy()
    span = model["span"].copy()

# Temsilî uzun ve kısa olaylar.
selected_ids = [
    "id_134", "id_116",
    "id_145", "id_18", "id_138",
    "id_122", "id_176", "id_160",
    "id_157", "id_153",
]

step_ns = pd.Timedelta("30s").value
rows = []

for event_id in selected_ids:
    records = labels[
        (labels["ID"] == event_id)
        & (labels["Category"] == "Anomaly")
    ]

    if records.empty:
        raise ValueError(f"{event_id}: olay kaydı bulunamadı.")

    mask = np.zeros(len(endpoints), dtype=bool)

    for record in records.itertuples(index=False):
        left = np.searchsorted(
            timestamps, record.StartTime.value, side="left"
        )
        right = np.searchsorted(
            timestamps,
            record.EndTime.value + step_ns,
            side="left",
        )
        mask[left:right] = True

    values = (
        (features[endpoints[mask]] - minimum) / span
    ).astype("float32")

    if len(values) == 0:
        raise ValueError(f"{event_id}: değerlendirme noktası yok.")

    for column in range(values.shape[1]):
        query = values[:, column]
        ref = reference[:, column]

        left_count = np.searchsorted(
            ref, query, side="right"
        )
        right_count = len(ref) - np.searchsorted(
            ref, query, side="left"
        )

        rows.append({
            "ID": event_id,
            "channel": f"channel_{41 + column}",
            "points": len(query),
            "below_fit_min_fraction": float(
                (query < ref[0]).mean()
            ),
            "above_fit_max_fraction": float(
                (query > ref[-1]).mean()
            ),
            "zero_empirical_tail_fraction": float(
                ((left_count == 0) | (right_count == 0)).mean()
            ),
            "distinct_scaled_values": len(np.unique(query)),
        })

report = pd.DataFrame(rows)

report.to_csv(
    OUTPUT / "ecod_tail_audit.csv",
    index=False,
)

print(report.to_string(index=False))