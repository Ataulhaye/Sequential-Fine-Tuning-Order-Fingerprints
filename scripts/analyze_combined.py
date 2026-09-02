import json
import sys
from pathlib import Path

# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from sequential_finetuning.combined_analysis import (
    combine_all_results,
)
from sequential_finetuning.config import load_config


def load_json(path: Path):
    """Load JSON from disk."""

    if not path.exists():
        raise FileNotFoundError(f"Required result file not found:\n{path}")

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config("configs/experiment.yaml")

    result_root = Path(config["paths"]["results"])

    sequential_orders = config["experiment"]["sequential_orders"]

    # --------------------------------------------------------
    # Result files
    # --------------------------------------------------------

    sequential_root = result_root / "sequential"

    weight_path = result_root / "weight_distance" / "weight_distance.json"

    representation_path = result_root / "representation" / "representation.json"

    loss_barrier_path = result_root / "loss_barrier" / "loss_barrier.json"

    jacobian_path = result_root / "jacobian" / "jacobian_analysis.json"

    # --------------------------------------------------------
    # Load existing analyses
    # --------------------------------------------------------

    print("=" * 70)
    print("COMBINED ANALYSIS")
    print("=" * 70)

    print()
    print("Loading sequential evaluation results...")
    sequential_results = {}

    # Sequential evaluation stores one metrics.json
    # inside each order directory.
    first_order_path = sequential_root / "_".join(sequential_orders[0]) / "metrics.json"

    if first_order_path.exists():
        sequential_results["orders"] = []

        for order in sequential_orders:

            order_name = "_".join(order)

            metrics_path = sequential_root / order_name / "metrics.json"

            data = load_json(metrics_path)

            sequential_results["orders"].append(data)

    else:
        raise FileNotFoundError("Sequential evaluation results were not found.")

    print(f"Loaded {len(sequential_results['orders'])} sequential results.")

    print()
    print("Loading weight-distance results...")
    weight_results = load_json(weight_path)

    print(f"Loaded {len(weight_results['orders'])} " "weight-distance results.")

    print()
    print("Loading representation results...")
    representation_results = load_json(representation_path)

    print(f"Loaded {len(representation_results['orders'])} " "representation results.")

    # --------------------------------------------------------
    # Optional: Load advanced diagnostics
    # --------------------------------------------------------

    loss_barrier_results = None
    jacobian_results = None

    if loss_barrier_path.exists():
        print()
        print("Loading loss-barrier results...")
        loss_barrier_results = load_json(loss_barrier_path)
        print(f"Loaded loss-barrier analysis for {len(loss_barrier_results)} orders.")
    else:
        print()
        print("Loss-barrier results not found (optional).")

    if jacobian_path.exists():
        print()
        print("Loading Jacobian sensitivity results...")
        jacobian_results = load_json(jacobian_path)
        print(f"Loaded Jacobian analysis for {len(jacobian_results)} orders.")
    else:
        print()
        print("Jacobian sensitivity results not found (optional).")

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    combined = combine_all_results(
        orders=sequential_orders,
        sequential_results=sequential_results,
        weight_results=weight_results,
        representation_results=representation_results,
        loss_barrier_results=loss_barrier_results,
        jacobian_results=jacobian_results,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("#" * 70)
    print("COMBINED ANALYSIS SUMMARY")
    print("#" * 70)

    total = len(combined["orders"])

    full_model_correct = 0
    backbone_correct = 0
    cka_correct = 0
    drift_correct = 0

    for result in combined["orders"]:

        order = result["order"]
        actual = result["actual_last_task"]

        full_model = result["weight_distance"]["full_model"]
        backbone = result["weight_distance"]["backbone"]

        cka = result["representation"]
        drift = result["representation"]

        full_model_correct += int(full_model["correct"])

        backbone_correct += int(backbone["correct"])

        cka_correct += int(cka["cka_correct"])

        drift_correct += int(drift["drift_correct"])

        print()
        print(f"Order: {' -> '.join(order)}")
        print(f"Actual last: {actual}")

        print(f"Weight distance: " f"{full_model['predicted_last_task']}")

        print(f"Backbone distance: " f"{backbone['predicted_last_task']}")

        print(f"CKA: " f"{cka['predicted_last_task']}")

        print(f"Feature drift: " f"{drift['drift_predicted_last_task']}")

    # --------------------------------------------------------
    # Prediction summary
    # --------------------------------------------------------

    print()
    print("-" * 70)
    print("LAST-TASK PREDICTION ACCURACY")
    print("-" * 70)

    print(
        f"Full-model weight distance: "
        f"{full_model_correct}/{total} "
        f"({100.0 * full_model_correct / total:.2f}%)"
    )

    print(
        f"Backbone weight distance:   "
        f"{backbone_correct}/{total} "
        f"({100.0 * backbone_correct / total:.2f}%)"
    )

    print(
        f"CKA:                        "
        f"{cka_correct}/{total} "
        f"({100.0 * cka_correct / total:.2f}%)"
    )

    print(
        f"Feature drift:              "
        f"{drift_correct}/{total} "
        f"({100.0 * drift_correct / total:.2f}%)"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = result_root / "combined"

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = output_dir / "combined_analysis.json"

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            combined,
            file,
            indent=2,
        )

    print()
    print("Results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
