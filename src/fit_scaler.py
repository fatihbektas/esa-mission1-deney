import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"

with (ROOT / "split_manifest.json").open() as file:
    protocol = json.load(file)

with (ROOT / "manifest.json").open() as file:
    manifest = json.load(file)

features = np.load(ROOT / "features.npy", mmap_mode="r")
eligible = np.load(ROOT / "eligible.npy", mmap_mode="r")

train = protocol["splits"]["train"]
start = train["start_index"]
stop = train["stop_index_exclusive"]

minimum = np.full(features.shape[1], np.inf, dtype="float64")
maximum = np.full(features.shape[1], -np.inf, dtype="float64")

used_rows = 0
chunk_size = 100_000

for left in range(start, stop, chunk_size):
    right = min(left + chunk_size, stop)

    block = np.asarray(features[left:right])
    mask = eligible[left:right]
    block = block[mask]

    if len(block) == 0:
        continue

    if not np.isfinite(block).all():
        raise ValueError("Eğitim verisinde geçersiz değer bulundu.")

    minimum = np.minimum(minimum, block.min(axis=0))
    maximum = np.maximum(maximum, block.max(axis=0))
    used_rows += len(block)

if used_rows == 0:
    raise ValueError("Ölçekleme için uygun eğitim satırı yok.")

span = maximum - minimum
constant = span == 0

# Sabit kanallarda sıfıra bölmeyi önle.
span[constant] = 1.0

np.savez(
    ROOT / "scaler.npz",
    minimum=minimum,
    maximum=maximum,
    span=span,
    constant_channels=constant,
)

metadata = {
    "method": "Min-Max",
    "formula": "(x - minimum) / span",
    "fit_split": "train",
    "fit_start_index": start,
    "fit_stop_index_exclusive": stop,
    "fit_rows": used_rows,
    "channels": manifest["channels"],
    "excluded": "rows marked ineligible by communication-gap policy",
    "anomaly_filter": False,
    "clip": False,
}

with (ROOT / "scaler_manifest.json").open(
    "w", encoding="utf-8"
) as file:
    json.dump(metadata, file, ensure_ascii=False, indent=2)

print(f"Kullanılan eğitim satırı: {used_rows:,}\n")

for channel, low, high, is_constant in zip(
    manifest["channels"], minimum, maximum, constant
):
    print(
        f"{channel}: min={low:.6g}, max={high:.6g}, "
        f"sabit={bool(is_constant)}"
    )

print("\nÖlçekleme istatistikleri kaydedildi.")