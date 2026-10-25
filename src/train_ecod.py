import json
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import skew


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "ecod"
OUTPUT.mkdir(parents=True, exist_ok=True)

# İlk hızlı deney için eğitimden etiketsiz rastgele alt örneklem.
MAX_FIT_ROWS = 100_000
SEED = 42

# Etiket kullanmadan belirlenen doğrulama skor yüzdeliği.
THRESHOLD_QUANTILE = 0.999

with (ROOT / "split_manifest.json").open() as file:
    protocol = json.load(file)

features = np.load(ROOT / "features.npy", mmap_mode="r")
eligible = np.load(ROOT / "eligible.npy", mmap_mode="r")

with np.load(ROOT / "scaler.npz") as scaler:
    minimum = scaler["minimum"]
    span = scaler["span"]


def transform(values):
    return ((values - minimum) / span).astype("float32")


def score_block(values, reference, skew_sign):
    """Eğitimde sabitlenmiş ECDF ile ECOD skorunu hesapla."""
    n = len(reference)
    scores = np.zeros(len(values), dtype="float64")

    for column in range(values.shape[1]):
        sorted_values = reference[:, column]
        query = values[:, column]

        # Eşit değerlere iki kuyrukta da kapsayıcı davran.
        left_probability = (
            np.searchsorted(sorted_values, query, side="right") / n
        )

        right_probability = (
            n - np.searchsorted(sorted_values, query, side="left")
        ) / n

        # Eğitim aralığı dışındaki değerlerde log(0) oluşmasını önle.
        probability_floor = 1.0 / (n + 1)

        left_tail = -np.log(
            np.clip(left_probability, probability_floor, 1)
        )
        right_tail = -np.log(
            np.clip(right_probability, probability_floor, 1)
        )

        skew_tail = (
            -left_tail * np.sign(skew_sign[column] - 1)
            + right_tail * np.sign(skew_sign[column] + 1)
        )

        scores += np.maximum(
            np.maximum(left_tail, right_tail),
            skew_tail,
        )

    return scores


started = time.perf_counter()

# 1. Eğitim satırlarını seç: anomali etiketleri okunmuyor.
train = protocol["splits"]["train"]
train_indices = np.flatnonzero(
    eligible[
        train["start_index"]:train["stop_index_exclusive"]
    ]
) + train["start_index"]

rng = np.random.default_rng(SEED)

if len(train_indices) > MAX_FIT_ROWS:
    train_indices = np.sort(
        rng.choice(
            train_indices,
            size=MAX_FIT_ROWS,
            replace=False,
        )
    )

train_values = transform(features[train_indices])

if not np.isfinite(train_values).all():
    raise ValueError("Geçersiz eğitim değeri bulundu.")

reference = np.sort(train_values, axis=0)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)
    skew_sign = np.sign(
        np.nan_to_num(skew(train_values, axis=0))
    )

np.save(OUTPUT / "fit_indices.npy", train_indices)

print(f"ECOD eğitim satırı: {len(train_indices):,}")

# 2. Doğrulamanın tamamını parçalar halinde puanla.
validation = protocol["splits"]["validation"]
start = validation["start_index"]
stop = validation["stop_index_exclusive"]

scores = np.lib.format.open_memmap(
    OUTPUT / "validation_scores.npy",
    mode="w+",
    dtype="float64",
    shape=(stop - start,),
)
scores[:] = np.nan

chunk_size = 100_000

for left in range(start, stop, chunk_size):
    right = min(left + chunk_size, stop)
    mask = eligible[left:right]

    block_scores = np.full(right - left, np.nan)

    if mask.any():
        values = transform(features[left:right][mask])
        block_scores[mask] = score_block(
            values, reference, skew_sign
        )

    scores[left - start:right - start] = block_scores

scores.flush()

valid_scores = np.asarray(scores)[np.isfinite(scores)]

if len(valid_scores) == 0:
    raise ValueError("Puanlanabilir doğrulama satırı yok.")

threshold = float(
    np.quantile(valid_scores, THRESHOLD_QUANTILE)
)
alarm_count = int((valid_scores > threshold).sum())

np.savez(
    OUTPUT / "model.npz",
    reference=reference,
    skew_sign=skew_sign,
    minimum=minimum,
    span=span,
    threshold=threshold,
)

report = {
    "model": "ECOD with training-frozen ECDF and skew",
    "fit_rows": len(train_indices),
    "seed": SEED,
    "training_labels_used": False,
    "validation_labels_used_for_threshold": False,
    "threshold_quantile": THRESHOLD_QUANTILE,
    "threshold": threshold,
    "comparison_operator": ">",
    "validation_rows_scored": len(valid_scores),
    "validation_alarm_points": alarm_count,
    "validation_alarm_fraction": alarm_count / len(valid_scores),
    "elapsed_seconds": time.perf_counter() - started,
    "test_scored": False,
}

with (OUTPUT / "run.json").open("w", encoding="utf-8") as file:
    json.dump(report, file, ensure_ascii=False, indent=2)

print(json.dumps(report, ensure_ascii=False, indent=2))