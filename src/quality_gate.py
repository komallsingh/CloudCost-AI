import json
import math
from pathlib import Path
from typing import Dict, Mapping


ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_REPORT_PATH = ROOT_DIR / "reports" / "quality_gate_decision.json"


def _normalize_metrics(metrics: Mapping[str, float]) -> Dict[str, float]:
    normalized = {}

    for name, value in metrics.items():
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Metric {name!r} must be finite")
        normalized[name] = number

    if "RMSE" not in normalized:
        raise ValueError("Metrics must include RMSE")

    if normalized["RMSE"] < 0:
        raise ValueError("RMSE must not be negative")

    return normalized


def run_quality_gate(
    champion_metrics: Mapping[str, float],
    candidate_metrics: Mapping[str, float],
    report_path: Path = DEFAULT_REPORT_PATH,
) -> Dict[str, object]:
    champion = _normalize_metrics(champion_metrics)
    candidate = _normalize_metrics(candidate_metrics)

    eligible = candidate["RMSE"] <= champion["RMSE"]
    report = {
        "decision": (
            "eligible_for_promotion" if eligible else "retain_champion"
        ),
        "candidate_eligible": eligible,
        "champion_updated": False,
        "comparison_metric": "RMSE",
        "criterion": "candidate RMSE <= champion RMSE",
        "justification": (
            "RMSE is the existing model-selection metric and penalizes "
            "larger CPU forecast errors more heavily."
        ),
        "champion_metrics": champion,
        "candidate_metrics": candidate,
    }

    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as report_file:
        json.dump(report, report_file, indent=4)

    print("\nQUALITY GATE")
    print("-" * 40)
    print("Decision:", report["decision"])
    print("Champion RMSE:", champion["RMSE"])
    print("Candidate RMSE:", candidate["RMSE"])
    print("Champion updated:", report["champion_updated"])
    print("Decision report saved:", report_path)

    return report
