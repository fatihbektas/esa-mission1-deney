import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

with (PROJECT_ROOT / "config.json").open() as file:
    config = json.load(file)


DATA_ROOT = PROJECT_ROOT / config["data_root"]
OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

CHANNELS = config["channels"]
FREQUENCY = config["frequency"]

# 
def read_channel(channel):
    """Kanalı oku, zamanları UTC yap ve sıralamayı düzenle."""
    path = DATA_ROOT / "channels" / f"{channel}.zip"

    frame = pd.read_pickle(path)
    
    timestamps = pd.to_datetime(frame.index, utc=True, errors="raise")
    values = pd.to_numeric(frame[channel], errors="raise").to_numpy()

    series = pd.Series(values, index=timestamps, name=channel)

    if series.empty or series.index.hasnans:
        raise ValueError(f"{channel}: boş veri veya eksik zaman damgası.")

    series = series.sort_index(kind="stable")

    # Aynı zamandaki kayıtlar için dosyadaki son değeri koru.
    duplicate_count = int(series.index.duplicated().sum())
    series = series[~series.index.duplicated(keep="last")]

    if not np.isfinite(series.to_numpy()).all():
        raise ValueError(f"{channel}: eksik veya sonsuz ölçüm bulundu.")

    return series, duplicate_count


# 1. Altı kanalın ortak zaman kapsamını bul.
bounds = []

for channel in CHANNELS:
    series, _ = read_channel(channel)
    bounds.append((series.index.min(), series.index.max()))

    del series

start = max(first for first, _ in bounds).ceil(FREQUENCY)
end = min(last for _, last in bounds).floor(FREQUENCY)

if start > end:
    raise ValueError("Kanalların ortak zaman aralığı yok.")

grid = pd.date_range(start, end, freq=FREQUENCY)
shape = (len(grid), len(CHANNELS))

# Büyük dizileri doğrudan diske yaz.
features = np.lib.format.open_memmap(
    OUTPUT_ROOT / "features.npy",
    mode="w+",
    dtype="float32",
    shape=shape,
)

ages = np.lib.format.open_memmap(
    OUTPUT_ROOT / "observation_age_seconds.npy",
    mode="w+",
    dtype="float32",
    shape=shape,
)

summary = []

# 2. Her kanalı ortak zaman eksenine ZOH ile hizala.
for column, channel in enumerate(CHANNELS):
    series, duplicate_count = read_channel(channel)

    # Her grid noktası için zaman <= grid olan son kaydı bul.
    source_positions = series.index.get_indexer(grid, method="pad")

    if (source_positions < 0).any():
        raise ValueError(f"{channel}: geçmiş ölçüm bulunamayan nokta var.")

    source_times = series.index.take(source_positions)
    aligned_values = series.to_numpy()[source_positions]

    age_seconds = (
        (grid - source_times) / pd.Timedelta(seconds=1)
    ).to_numpy()

    if (age_seconds < 0).any():
        raise ValueError(f"{channel}: gelecekteki ölçüm kullanılmış.")

    values32 = aligned_values.astype("float32")

    if not np.isfinite(values32).all():
        raise ValueError(f"{channel}: float32 dönüşümü geçersiz değer üretti.")

    features[:, column] = values32
    ages[:, column] = age_seconds.astype("float32")

    summary.append({
        "channel": channel,
        "raw_rows_after_deduplication": len(series),
        "duplicate_timestamps_removed": duplicate_count,
        "aligned_rows": len(grid),
        "median_age_seconds": float(np.median(age_seconds)),
        "max_age_seconds": float(age_seconds.max()),
        "age_over_60_seconds": int((age_seconds > 60).sum()),
    })

    print(f"{channel}: {len(series):,} → {len(grid):,} nokta")

    del series, aligned_values, values32, source_times, age_seconds

features.flush()
ages.flush()

# Açık zaman birimiyle UTC eksenini kaydet.
timestamps_ns = grid.to_numpy(dtype="datetime64[ns]").astype("int64")
np.save(OUTPUT_ROOT / "timestamps_ns.npy", timestamps_ns)

manifest = {
    "channels": CHANNELS,
    "frequency": FREQUENCY,
    "start_utc": str(start),
    "end_utc": str(end),
    "shape": list(shape),
    "timestamp_unit": "nanoseconds since Unix epoch, UTC",
    "feature_dtype": "float32",
    "method": "ZOH: last observation at or before grid timestamp",
    "duplicate_policy": "keep last",
    "gap_policy": "unlimited forward hold; observation age recorded",
    "summary": summary,
}

with (OUTPUT_ROOT / "manifest.json").open("w", encoding="utf-8") as file:
    json.dump(manifest, file, ensure_ascii=False, indent=2)

pd.DataFrame(summary).to_csv(
    OUTPUT_ROOT / "preprocessing_summary.csv",
    index=False,
)

print(f"\nTamamlandı. Matris boyutu: {shape}")
print(f"Başlangıç: {start}")
print(f"Bitiş: {end}")
print(f"Çıktı: {OUTPUT_ROOT}")