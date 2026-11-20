
import pandas as pd
import json
from pathlib import Path

DATA_PATH = Path("data/new_cpu_batch.csv")
REPORT_DIR = Path("reports")

REQUIRED_COLUMNS = [
    "timestamp",
    "vm_id",
    "min_cpu",
    "max_cpu",
    "avg_cpu",
    "subscription_id",
    "deployment_id",
    "timestamp_vm_created",
    "timestamp_vm_deleted",
    "vm_metadata_max_cpu",
    "vm_metadata_avg_cpu",
    "p95_max_cpu",
    "vm_category",
    "virtual_core_bucket",
    "memory_gb_bucket"
]


def validate_new_data():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    # Check required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Check missing values
    missing_values = df.isnull().sum().to_dict()

    # Check duplicate records
    duplicate_records = int(df.duplicated().sum())

    # Check duplicate VM and timestamp combinations
    duplicate_ids = int(
        df.duplicated(
            subset=["vm_id", "timestamp"]
        ).sum()
    )

    # Validate CPU values
    if not df["avg_cpu"].between(0, 100).all():
        raise ValueError("Invalid avg_cpu values")

    if not (
        (df["min_cpu"] <= df["avg_cpu"]) &
        (df["avg_cpu"] <= df["max_cpu"])
    ).all():
        raise ValueError("Invalid CPU min/average/max relationship")

    report = {
        "dataset": str(DATA_PATH),
        "total_records": len(df),
        "total_columns": len(df.columns),
        "missing_values": missing_values,
        "duplicate_records": duplicate_records,
        "duplicate_vm_timestamp": duplicate_ids,
        "target_distribution": df["avg_cpu"].describe().to_dict(),
        "status": "VALIDATED"
    }

    with open(
        REPORT_DIR / "data_validation_report.json",
        "w"
    ) as f:
        json.dump(report, f, indent=4)

    print("\nDATA VALIDATION RESULTS")
    print("-----------------------")
    print("Records:", len(df))
    print("Columns:", len(df.columns))
    print("Duplicate records:", duplicate_records)
    print("Duplicate VM timestamps:", duplicate_ids)
    print("\nCPU distribution:")
    print(df["avg_cpu"].describe())

    return df


if __name__ == "__main__":
    validate_new_data()
