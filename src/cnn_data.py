import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"
WINDOW = 100


def window_endpoints(eligible, start, stop, window=WINDOW):
    """Tamamı uygun ve aynı bölümde olan pencerelerin son satırları."""
    if stop - start < window:
        return np.empty(0, dtype="int64")

    invalid = ~np.asarray(eligible[start:stop], dtype=bool)

    cumulative = np.empty(len(invalid) + 1, dtype="int64")
    cumulative[0] = 0
    np.cumsum(invalid, out=cumulative[1:])

    invalid_counts = (
        cumulative[window:] - cumulative[:-window]
    )

    return (
        np.flatnonzero(invalid_counts == 0)
        + start
        + window
        - 1
    )


class TelemetryWindows(Dataset):
    def __init__(self, endpoints, window=WINDOW):
        self.endpoints = np.asarray(endpoints, dtype="int64")
        self.window = window

        self.features = np.load(
            ROOT / "features.npy", mmap_mode="r"
        )
        self.target = np.load(
            ROOT / "target.npy", mmap_mode="r"
        )

        with np.load(ROOT / "scaler.npz") as scaler:
            self.minimum = scaler["minimum"].copy()
            self.span = scaler["span"].copy()

    def __len__(self):
        return len(self.endpoints)

    def __getitem__(self, index):
        end = int(self.endpoints[index])
        start = end - self.window + 1

        values = self.features[start:end + 1]

        scaled = (
            (values - self.minimum) / self.span
        ).astype("float32")

        # PyTorch Conv1d: [kanal, zaman]
        inputs = torch.from_numpy(scaled.T.copy())
        target = torch.tensor(
            float(self.target[end]),
            dtype=torch.float32,
        )

        return inputs, target


def main():
    with (ROOT / "split_manifest.json").open() as file:
        protocol = json.load(file)

    eligible = np.load(ROOT / "eligible.npy", mmap_mode="r")
    target = np.load(ROOT / "target.npy", mmap_mode="r")

    # Test pencerelerini şimdilik hazırlamıyoruz.
    for name in ["train", "validation"]:
        section = protocol["splits"][name]

        endpoints = window_endpoints(
            eligible,
            section["start_index"],
            section["stop_index_exclusive"],
        )

        if len(endpoints) == 0:
            raise ValueError(f"{name}: uygun pencere yok.")

        np.save(ROOT / f"{name}_window_endpoints.npy", endpoints)

        positives = int(target[endpoints].sum())

        print(
            f"{name}: {len(endpoints):,} pencere, "
            f"{positives:,} pozitif hedef, "
            f"oran={positives / len(endpoints):.6f}"
        )

        dataset = TelemetryWindows(endpoints)

        for position in [0, len(dataset) - 1]:
            inputs, label = dataset[position]

            assert inputs.shape == (6, WINDOW)
            assert torch.isfinite(inputs).all()
            assert label.item() in (0.0, 1.0)

        print("  İlk ve son pencere kontrolleri geçti.")


if __name__ == "__main__":
    main()