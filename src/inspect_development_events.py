from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ROOT = PROJECT_ROOT / "data" / "processed"

labels = pd.read_csv(ROOT / "selected_label_intervals.csv")

for column in ["StartTime", "EndTime"]:
    labels[column] = pd.to_datetime(labels[column], utc=True)

development_start = pd.Timestamp("2000-01-01", tz="UTC")
test_start = pd.Timestamp("2007-01-01", tz="UTC")

# Yalnızca test öncesi gerçek anomaliler.
events = labels[
    (labels["Category"] == "Anomaly")
    & (labels["StartTime"] < test_start)
    & (labels["EndTime"] >= development_start)
].copy()

# Aynı ID'nin farklı kanallardaki kayıtlarını bir olayda topla.
events = (
    events.groupby("ID", as_index=False)
    .agg(
        start=("StartTime", "min"),
        end=("EndTime", "max"),
        channel_count=("Channel", "nunique"),
    )
    .sort_values("start")
)

events["start_year"] = events["start"].dt.year

print("Test öncesi benzersiz anomali olayları:\n")
print(events.to_string(index=False))

print("\nBaşlangıç yılına göre olay sayıları:\n")
print(events.groupby("start_year").size().to_string())

events.to_csv(
    ROOT / "development_event_inventory.csv",
    index=False,
)