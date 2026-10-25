import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"

with (PROJECT_ROOT / "config.json").open() as file:
    config = json.load(file)

with (ROOT / "manifest.json").open() as file:
    manifest = json.load(file)

if manifest["channels"] != config["channels"]:
    raise ValueError("Ön işleme ile config kanal sıraları farklı.")

timestamps = np.load(ROOT / "timestamps_ns.npy", mmap_mode="r")
features = np.load(ROOT / "features.npy", mmap_mode="r")

anomaly = np.load(ROOT / "anomaly_mask.npy", mmap_mode="r")
rare = np.load(ROOT / "rare_event_mask.npy", mmap_mode="r")
gap = np.load(ROOT / "communication_gap_mask.npy", mmap_mode="r")

expected_shape = (len(timestamps), len(config["channels"]))

for name, array in [
    ("features", features),
    ("anomaly", anomaly),
    ("rare_event", rare),
    ("communication_gap", gap),
]:
    if array.shape != expected_shape:
        raise ValueError(f"{name}: beklenmeyen boyut {array.shape}")

val_start = pd.Timestamp("2005-01-01", tz="UTC").value
test_start = pd.Timestamp("2007-01-01", tz="UTC").value

val_position = int(np.searchsorted(timestamps, val_start))
test_position = int(np.searchsorted(timestamps, test_start))

# Başlangıç dahil, bitiş hariç satır aralıkları.
splits = {
    "train": (0, val_position),
    "validation": (val_position, test_position),
    "test": (test_position, len(timestamps)),
}

# Tek bir çok değişkenli alarm hedefi:
# Altı kanaldan herhangi birinde gerçek anomali varsa pozitif.
target = anomaly.any(axis=1)

# Herhangi bir kanalda etiketli iletişim boşluğu varsa değerlendirme dışı.
eligible = ~gap.any(axis=1)

# Rare Event tek başına pozitif hedef değildir.
# Anomaly ile çakışıyorsa pozitif hedef korunur.
rare_any = rare.any(axis=1)

np.save(ROOT / "target.npy", target.astype("uint8"))
np.save(ROOT / "eligible.npy", eligible)

summary = []

for name, (start, stop) in splits.items():
    if start >= stop:
        raise ValueError(f"{name}: boş bölüm.")

    section_target = target[start:stop]
    section_eligible = eligible[start:stop]

    positive_count = int(
        (section_target & section_eligible).sum()
    )
    eligible_count = int(section_eligible.sum())

    summary.append({
        "split": name,
        "start_utc": str(
            pd.to_datetime(timestamps[start], unit="ns", utc=True)
        ),
        "last_timestamp_utc": str(
            pd.to_datetime(timestamps[stop - 1], unit="ns", utc=True)
        ),
        "rows": stop - start,
        "eligible_rows": eligible_count,
        "excluded_gap_rows": int((~section_eligible).sum()),
        "anomaly_rows": positive_count,
        "rare_only_rows": int((
            rare_any[start:stop]
            & ~section_target
            & section_eligible
        ).sum()),
        "anomaly_fraction": (
            positive_count / eligible_count
            if eligible_count else None
        ),
    })

protocol = {
    "splits": {
        name: {
            "start_index": start,
            "stop_index_exclusive": stop,
        }
        for name, (start, stop) in splits.items()
    },
    "validation_start_utc": "2005-01-01T00:00:00Z",
    "test_start_utc": "2007-01-01T00:00:00Z",
    "target": "any selected channel has Anomaly",
    "rare_event_policy": "negative unless overlapping Anomaly",
    "communication_gap_policy": "exclude if any channel has gap",
    "label_time_semantics": "(t - 30 seconds, t]",
    "observation_age_filter": None,
    "summary": summary,
}

with (ROOT / "split_manifest.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(protocol, file, ensure_ascii=False, indent=2)

report = pd.DataFrame(summary)
report.to_csv(ROOT / "split_summary.csv", index=False)

print(report.to_string(index=False))
print("\nHedef, uygunluk maskesi ve bölüm sınırları kaydedildi.")