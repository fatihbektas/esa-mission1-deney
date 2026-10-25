import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from cnn_data import ROOT, WINDOW, TelemetryWindows


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "results" / "cnn"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 42
MAX_TRAIN_WINDOWS = 100_000
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001
THRESHOLD_QUANTILE = 0.999


class MultiscaleCNN(nn.Module):
    def __init__(self, window=WINDOW):
        super().__init__()

        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(
                    in_channels=6,
                    out_channels=64,
                    kernel_size=kernel,
                    padding=kernel // 2,
                ),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
            )
            for kernel in [3, 5, 7]
        ])

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192 * (window // 2), 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )

    def forward(self, inputs):
        outputs = [branch(inputs) for branch in self.branches]
        combined = torch.cat(outputs, dim=1)

        # Logit döndür; sigmoid kayıp fonksiyonunda uygulanır.
        return self.classifier(combined).squeeze(1)


def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)

    # İlk deneyde tutarlı ve kolay denetlenebilir CPU çalışması.
    device = torch.device("cpu")
    started = time.perf_counter()

    train_endpoints = np.load(
        ROOT / "train_window_endpoints.npy"
    )
    validation_endpoints = np.load(
        ROOT / "validation_window_endpoints.npy"
    )

    rng = np.random.default_rng(SEED)

    if len(train_endpoints) > MAX_TRAIN_WINDOWS:
        train_endpoints = np.sort(
            rng.choice(
                train_endpoints,
                size=MAX_TRAIN_WINDOWS,
                replace=False,
            )
        )

    target = np.load(ROOT / "target.npy", mmap_mode="r")
    positive_count = int(target[train_endpoints].sum())

    if positive_count == 0 or positive_count == len(train_endpoints):
        raise ValueError("Eğitim örnekleminde iki sınıf da bulunmalı.")

    np.save(OUTPUT / "fit_endpoints.npy", train_endpoints)

    train_dataset = TelemetryWindows(train_endpoints)
    validation_dataset = TelemetryWindows(validation_endpoints)

    generator = torch.Generator().manual_seed(SEED)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = MultiscaleCNN().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )
    criterion = nn.BCEWithLogitsLoss()

    print(
        f"Eğitim: {len(train_dataset):,} pencere, "
        f"{positive_count:,} pozitif",
        flush=True,
    )

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        loss_sum = 0.0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(inputs)
            loss = criterion(logits, labels)

            if not torch.isfinite(loss):
                raise ValueError("Sonlu olmayan eğitim kaybı.")

            loss.backward()
            optimizer.step()

            loss_sum += loss.item() * len(labels)

        mean_loss = loss_sum / len(train_dataset)
        history.append({
            "epoch": epoch,
            "train_loss": mean_loss,
        })

        print(
            f"Epoch {epoch}/{EPOCHS}: loss={mean_loss:.6f}",
            flush=True,
        )

    # Eğitim tamamlandığında modeli hemen kaydet.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "window": WINDOW,
            "channels": 6,
        },
        OUTPUT / "model.pt",
    )

    # Doğrulama skorları endpoints dizisiyle aynı sırada yazılır.
    scores = np.lib.format.open_memmap(
        OUTPUT / "validation_scores.npy",
        mode="w+",
        dtype="float32",
        shape=(len(validation_dataset),),
    )

    model.eval()
    offset = 0

    with torch.inference_mode():
        for batch_number, (inputs, _) in enumerate(
            validation_loader, start=1
        ):
            probabilities = torch.sigmoid(
                model(inputs.to(device))
            ).cpu().numpy()

            scores[offset:offset + len(probabilities)] = probabilities
            offset += len(probabilities)

            if batch_number % 1000 == 0:
                print(
                    f"Doğrulama: {offset:,}/"
                    f"{len(validation_dataset):,}",
                    flush=True,
                )

    scores.flush()

    if not np.isfinite(scores).all():
        raise ValueError("Geçersiz doğrulama skoru.")

    threshold = float(
        np.quantile(scores, THRESHOLD_QUANTILE)
    )

    alarm_count = int((scores > threshold).sum())

    # Skorların hangi özgün satırlara ait olduğunu koru.
    np.save(
        OUTPUT / "validation_endpoints.npy",
        validation_endpoints,
    )

    report = {
        "model": "Multiscale CNN",
        "window": WINDOW,
        "kernels": [3, 5, 7],
        "filters_per_branch": 64,
        "hidden_units": 64,
        "dropout": 0.3,
        "fit_windows": len(train_dataset),
        "fit_positive_windows": positive_count,
        "seed": SEED,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "loss": "unweighted BCEWithLogitsLoss",
        "device": str(device),
        "threshold_quantile": THRESHOLD_QUANTILE,
        "threshold": threshold,
        "comparison_operator": ">",
        "validation_labels_used_for_threshold": False,
        "validation_windows_scored": len(validation_dataset),
        "validation_alarm_points": alarm_count,
        "validation_alarm_fraction": (
            alarm_count / len(validation_dataset)
        ),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "test_scored": False,
    }

    with (OUTPUT / "run.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()