
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature

from src.preprocessing import (
    create_features,
    temporal_split,
    FEATURES,
    TARGET
)
from src.quality_gate import run_quality_gate

from train import get_models

ROOT_DIR = Path(__file__).resolve().parent.parent

OLD_DATA_PATH = ROOT_DIR / "data" / "processed" / "vm_cpu_subset.csv"
NEW_DATA_PATH = ROOT_DIR / "data" / "incoming" / "new_cpu_batch.csv"

CANDIDATE_PATH = ROOT_DIR / "models" / "candidate.joblib"
CHAMPION_PATH = ROOT_DIR / "models" / "champion.joblib"
REPORT_DIR = ROOT_DIR / "reports"

MLFLOW_DB = ROOT_DIR / "mlflow.db"
TRACKING_URI = f"sqlite:///{MLFLOW_DB.as_posix()}"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_registry_uri(TRACKING_URI)
mlflow.set_experiment("CloudCost-AI-CPU-Forecasting")


def calculate_metrics(y_true, predictions):
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score
    )

    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "R2": float(r2_score(y_true, predictions))
    }


def retrain_model():

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    old_data = pd.read_csv(OLD_DATA_PATH)
    new_data = pd.read_csv(NEW_DATA_PATH)

    # Combine datasets
    combined_data = pd.concat(
        [old_data, new_data],
        ignore_index=True
    )

    # Remove duplicate VM timestamp records
    combined_data = combined_data.drop_duplicates(
        ["vm_id", "timestamp"],
        keep="last"
    )

    # Sort chronologically per VM
    combined_data = combined_data.sort_values(
        ["vm_id", "timestamp"]
    )

    # Feature engineering
    featured_data = create_features(combined_data)

    # Temporal train/test split
    X_train, X_test, y_train, y_test = temporal_split(featured_data)

    results = []
    trained_models = {}

    # Train candidate models
    for model_name, model in get_models().items():

        with mlflow.start_run(run_name=f"Retraining_{model_name}") as run:

            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            metrics = calculate_metrics(y_test, predictions)

            mlflow.log_params({
                "model_name": model_name,
                "training_rows": len(X_train),
                "testing_rows": len(X_test),
                "features": len(FEATURES),
                "purpose": "retraining"
            })

            mlflow.log_metrics(metrics)

            signature = infer_signature(
                X_train,
                model.predict(X_train.head(5))
            )

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                signature=signature,
                input_example=X_train.head(5)
            )

            trained_models[model_name] = model

            results.append({
                "model": model_name,
                "RMSE": metrics["RMSE"],
                "MAE": metrics["MAE"],
                "R2": metrics["R2"],
                "run_id": run.info.run_id
            })

            print(f"\n{model_name}")
            print("-" * 30)
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")

    # Select best candidate based on RMSE
    best_result = min(results, key=lambda item: item["RMSE"])

    best_model = trained_models[best_result["model"]]

    # Save candidate separately
    CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, CANDIDATE_PATH)

    champion = joblib.load(CHAMPION_PATH)
    champion_metrics = calculate_metrics(
        y_test,
        champion.predict(X_test)
    )
    candidate_metrics = {
        metric: best_result[metric]
        for metric in ("MAE", "RMSE", "R2")
    }
    best_result["quality_gate"] = run_quality_gate(
        champion_metrics,
        candidate_metrics,
        report_path=REPORT_DIR / "quality_gate_decision.json"
    )

    print("\nRETRAINING COMPLETE")
    print("-" * 40)
    print("Best model:", best_result["model"])
    print("Best RMSE:", best_result["RMSE"])
    print("Candidate saved:", CANDIDATE_PATH)

    return best_result


if __name__ == "__main__":
    retrain_model()
