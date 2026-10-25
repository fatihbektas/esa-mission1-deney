import json
from pathlib import Path

import numpy as np

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
OUTPUT = PROJECT_ROOT / "results" / "ecod"

with (ROOT / "split_manifest.json").open() as file:
    protocol = json.load(file)

features = np.load(ROOT / "features.npy", mmap_mode="r")
target = np.load(ROOT / "target.npy", mmap_mode="r")
eligible = np.load(ROOT / "eligible.npy", mmap_mode="r")
saved = np.load(
    OUTPUT / "validation_scores.npy", mmap_mode="r"
)

section = protocol["splits"]["validation"]
start = section["start_index"]
stop = section["stop_index_exclusive"]

# Eğitim indekslerinin doğrulama bölümüne taşmadığını kontrol et.
fit_indices = np.load(OUTPUT / "fit_indices.npy")
train = protocol["splits"]["train"]

assert (fit_indices >= train["start_index"]).all()
assert (fit_indices < train["stop_index_exclusive"]).all()

with np.load(OUTPUT / "model.npz") as model:
    reference = model["reference"]
    skew_sign = model["skew_sign"]
    minimum = model["minimum"]
    span = model["span"]

# Doğrulamadan rastgele satırları yeniden puanla.
rng = np.random.default_rng(123)
local_indices = np.sort(
    rng.choice(stop - start, size=10_000, replace=False)
)

local_indices = local_indices[
    eligible[start + local_indices]
]

values = (
    (features[start + local_indices] - minimum) / span
).astype("float32")

recomputed = score_block(values, reference, skew_sign)

np.testing.assert_allclose(
    recomputed,
    saved[local_indices],
    rtol=1e-10,
    atol=1e-10,
)

print("Eğitim indeksleri ve yeniden puanlama kontrolü geçti.")

# Sınıfların skor dağılımlarını karşılaştır.
y = target[start:stop].astype(bool)
valid = eligible[start:stop] & np.isfinite(saved)

quantiles = [0, 0.5, 0.9, 0.99, 0.999, 1]

for name, mask in [
    ("Normal", valid & ~y),
    ("Anomali", valid & y),
]:
    values = np.asarray(saved)[mask]
    print(f"\n{name}: {len(values):,} nokta")

    for q, value in zip(
        quantiles, np.quantile(values, quantiles)
    ):
        print(f"  q={q:5.3f}: {value:.6f}")