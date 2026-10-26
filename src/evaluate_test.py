import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader

from cnn_data import (
    ROOT,
    WINDOW,
    TelemetryWindows,
    window_endpoints,
)
from train_cnn import MultiscaleCNN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT_ROOT / "results"
OUTPUT = RESULTS / "test"
OUTPUT.mkdir(parents=True, exist_ok=True)


def ecod_scores(values, reference, skew_sign):
    """Eğitimde sabitlenmiş ECOD dağılımıyla puanla."""
    n = len(reference)
    scores = np.zeros(len(values), dtype="float64")

    for column in range(values.shape[1]):
        ref = reference[:, column]
        query = values[:, column]

        left = np.searchsorted(ref, query, side="right") / n
        right = (
            n - np.searchsorted(ref, query, side="left")
        ) / n

        floor = 1.0 / (n + 1)
        ul = -np.log(np.clip(left, floor, 1))
        ur = -np.log(np.clip(right, floor, 1))

        us = (
            -ul * np.sign(skew_sign[column] - 1)
            + ur * np.sign(skew_sign[column] + 1)
        )

        scores += np.maximum(np.maximum(ul, ur), us)

    return scores


def main():
    torch.set_num_threads(2)

    with (ROOT / "split_manifest.json").open() as file:
        protocol = json.load(file)

    with (ROOT / "manifest.json").open() as file:
        manifest = json.load(file)

    section = protocol["splits"]["test"]
    start = section["start_index"]
    stop = section["stop_index_exclusive"]

    eligible = np.load(ROOT / "eligible.npy", mmap_mode="r")

    endpoints = window_endpoints(
        eligible, start, stop, window=WINDOW
    )

    if len(endpoints) == 0:
        raise ValueError("Testte uygun pencere yok.")

    np.save(OUTPUT / "endpoints.npy", endpoints)

    timestamps = np.load(
        ROOT / "timestamps_ns.npy", mmap_mode="r"
    )[endpoints]

    target = np.load(
        ROOT / "target.npy", mmap_mode="r"
    )[endpoints].astype(bool)

    features = np.load(ROOT / "features.npy", mmap_mode="r")

    print(f"Ortak test noktası: {len(endpoints):,}", flush=True)

    # 1. ECOD: eğitimde kaydedilmiş dağılım ve ölçekleme.
    started = time.perf_counter()

    with np.load(RESULTS / "ecod" / "model.npz") as saved:
        reference = saved["reference"].copy()
        skew_sign = saved["skew_sign"].copy()
        minimum = saved["minimum"].copy()
        span = saved["span"].copy()
        ecod_threshold = float(saved["threshold"])

    ecod_output = np.lib.format.open_memmap(
        OUTPUT / "ecod_scores.npy",
        mode="w+",
        dtype="float64",
        shape=(len(endpoints),),
    )

    chunk_size = 100_000

    for left in range(0, len(endpoints), chunk_size):
        right = min(left + chunk_size, len(endpoints))
        indices = endpoints[left:right]

        values = (
            (features[indices] - minimum) / span
        ).astype("float32")

        ecod_output[left:right] = ecod_scores(
            values, reference, skew_sign
        )

    ecod_output.flush()
    ecod_seconds = time.perf_counter() - started

    print(
        f"ECOD puanlama tamamlandı: {ecod_seconds:.1f} sn",
        flush=True,
    )

    # 2. CNN: kayıtlı son epoch modeli, yeniden eğitim yok.
    started = time.perf_counter()

    with (RESULTS / "cnn" / "run.json").open() as file:
        cnn_run = json.load(file)

    checkpoint = torch.load(
        RESULTS / "cnn" / "model.pt",
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["window"] != WINDOW:
        raise ValueError("Model ve veri pencere uzunluğu farklı.")

    model = MultiscaleCNN(window=WINDOW)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    cnn_threshold = cnn_run["threshold"]

    loader = DataLoader(
        TelemetryWindows(endpoints),
        batch_size=cnn_run["batch_size"],
        shuffle=False,
        num_workers=0,
    )

    cnn_output = np.lib.format.open_memmap(
        OUTPUT / "cnn_scores.npy",
        mode="w+",
        dtype="float32",
        shape=(len(endpoints),),
    )

    offset = 0

    with torch.inference_mode():
        for batch_number, (inputs, _) in enumerate(loader, start=1):
            scores = torch.sigmoid(model(inputs)).numpy()

            cnn_output[offset:offset + len(scores)] = scores
            offset += len(scores)

            if batch_number % 1000 == 0:
                print(
                    f"CNN test: {offset:,}/{len(endpoints):,}",
                    flush=True,
                )

    cnn_output.flush()
    cnn_seconds = time.perf_counter() - started

    # 3. Özgün test olaylarını oku.
    labels = pd.read_csv(
        ROOT / "selected_label_intervals.csv"
    )

    for column in ["StartTime", "EndTime"]:
        labels[column] = pd.to_datetime(
            labels[column], utc=True
        )

    test_start = pd.Timestamp(protocol["test_start_utc"])
    step = pd.Timedelta(manifest["frequency"])

    # Mevcut son noktanın temsil ettiği kutuya kadar.
    test_end = pd.to_datetime(
        timestamps[-1], unit="ns", utc=True
    )

    events = labels[
        (labels["Category"] == "Anomaly")
        & (labels["StartTime"] <= test_end)
        & (labels["EndTime"] >= test_start)
    ]

    metric_rows = []
    event_rows = []

    for name, scores, threshold, seconds in [
        ("ecod", ecod_output, ecod_threshold, ecod_seconds),
        ("cnn", cnn_output, cnn_threshold, cnn_seconds),
    ]:
        if not np.isfinite(scores).all():
            raise ValueError(f"{name}: geçersiz test skoru.")

        prediction = scores > threshold

        tn, fp, fn, tp = confusion_matrix(
            target, prediction, labels=[False, True]
        ).ravel()

        continuous = np.r_[
            False,
            np.diff(timestamps) == step.value,
        ]
        previous_alarm = np.r_[False, prediction[:-1]]

        episodes = int((
            prediction & ~(previous_alarm & continuous)
        ).sum())

        evaluable_count = 0
        detected_count = 0

        for event_id, records in events.groupby("ID"):
            event_mask = np.zeros(len(endpoints), dtype=bool)

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

            hits = np.flatnonzero(event_mask & prediction)
            evaluable = bool(event_mask.any())
            detected = bool(len(hits))

            evaluable_count += int(evaluable)
            detected_count += int(detected)

            event_rows.append({
                "model": name,
                "ID": event_id,
                "start": str(records["StartTime"].min()),
                "end": str(records["EndTime"].max()),
                "evaluable_points": int(event_mask.sum()),
                "alarm_points": len(hits),
                "detected": detected,
                "first_alarm_utc": (
                    str(pd.to_datetime(
                        timestamps[hits[0]], unit="ns", utc=True
                    ))
                    if detected else None
                ),
            })

        metric_rows.append({
            "model": name,
            "evaluated_points": len(target),
            "threshold": threshold,
            "precision": precision_score(
                target, prediction, zero_division=0
            ),
            "recall": recall_score(
                target, prediction, zero_division=0
            ),
            "point_f0_5": fbeta_score(
                target, prediction, beta=0.5, zero_division=0
            ),
            "average_precision": average_precision_score(
                target, scores
            ),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "alarm_points": int(prediction.sum()),
            "alarm_fraction": float(prediction.mean()),
            "alarm_episodes": episodes,
            "evaluable_event_ids": evaluable_count,
            "detected_event_ids": detected_count,
            "scoring_seconds": seconds,
        })

    comparison = pd.DataFrame(metric_rows)
    event_table = pd.DataFrame(event_rows)

    comparison.to_csv(
        OUTPUT / "comparison.csv", index=False
    )
    event_table.to_csv(
        OUTPUT / "event_comparison.csv", index=False
    )

    with (OUTPUT / "protocol.json").open("w") as file:
        json.dump({
            "thresholds_frozen_from_validation": True,
            "test_threshold_optimization": False,
            "window": WINDOW,
            "common_endpoints": True,
            "event_matching": "(t - 30 seconds, t] bins",
            "official_esa_event_metrics": False,
        }, file, indent=2)

    print("\nTest karşılaştırması:\n")
    print(comparison.to_string(index=False))

    print("\nOlay başına alarm noktası:\n")
    print(event_table.pivot(
        index="ID",
        columns="model",
        values="alarm_points",
    ).to_string())


if __name__ == "__main__":
    main()