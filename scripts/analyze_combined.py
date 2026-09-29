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

    probe_enabled = config["representation"]["probe"]["enabled"]
    if not isinstance(probe_enabled, bool):
        raise ValueError("representation.probe.enabled must be true or false.")
    representation_mode = "probe" if probe_enabled else "all_test"
    representation_path = (
        result_root / "representation" / representation_mode / "representation.json"
    )
    if not representation_path.exists() and representation_mode == "probe":
        representation_path = result_root / "representation" / "representation.json"

    loss_barrier_path = result_root / "loss_barrier" / "loss_barrier.json"

    jacobian_path = result_root / "jacobian" / "jacobian_analysis.json"

    fisher_path = result_root / "fisher" / "fisher.json"

    weight_matching_path = result_root / "weight_matching" / "weight_matching.json"

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

    loss_barrier_enabled = config["analysis"]["loss_barrier"].get(
        "enabled",
        False,
    )

    jacobian_enabled = config["analysis"]["jacobian"].get(
        "enabled",
        False,
    )

    if loss_barrier_enabled:
        if not loss_barrier_path.exists():
            raise FileNotFoundError(
                "Loss-barrier analysis is enabled, but the result file "
                f"was not found: {loss_barrier_path}"
            )

        print()
        print("Loading loss-barrier results...")

        loss_barrier_results = load_json(loss_barrier_path)

        print(
            f"Loaded loss-barrier analysis for " f"{len(loss_barrier_results)} orders."
        )
    else:
        print()
        print("Loss-barrier analysis disabled in configuration.")

    if jacobian_enabled:
        if not jacobian_path.exists():
            raise FileNotFoundError(
                "Jacobian analysis is enabled, but the result file "
                f"was not found: {jacobian_path}"
            )

        print()
        print("Loading Jacobian sensitivity results...")

        jacobian_results = load_json(jacobian_path)

        print(f"Loaded Jacobian analysis for " f"{len(jacobian_results)} orders.")
    else:
        print()
        print("Jacobian analysis disabled in configuration.")

    fisher_results = None
    weight_matching_results = None

    fisher_enabled = config["analysis"].get("fisher", {}).get("enabled", False)

    weight_matching_enabled = (
        config["analysis"].get("weight_matching", {}).get("enabled", False)
    )

    if fisher_enabled:
        if not fisher_path.exists():
            raise FileNotFoundError(
                "Fisher analysis is enabled, but the result file "
                f"was not found: {fisher_path}"
            )

        print()
        print("Loading Fisher information results...")

        fisher_results = load_json(fisher_path)

        print(f"Loaded {len(fisher_results['orders'])} Fisher results.")
    else:
        print()
        print("Fisher analysis disabled in configuration.")

    if weight_matching_enabled:
        if not weight_matching_path.exists():
            raise FileNotFoundError(
                "Weight-matching analysis is enabled, but the result file "
                f"was not found: {weight_matching_path}"
            )

        print()
        print("Loading weight-matching results...")

        weight_matching_results = load_json(weight_matching_path)

        print(
            f"Loaded {len(weight_matching_results['orders'])} "
            "weight-matching results."
        )
    else:
        print()
        print("Weight-matching analysis disabled in configuration.")

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
        fisher_results=fisher_results,
        weight_matching_results=weight_matching_results,
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

    if fisher_results is not None:
        print()
        print("-" * 70)
        print("FISHER INFORMATION")
        print("-" * 70)

        for metric, hypotheses in fisher_results["summary"]["metrics"].items():
            for hypothesis, values in hypotheses.items():
                print(
                    f"{metric} / {hypothesis}: "
                    f"full order {values['correct_orders']}/{total} "
                    f"({100.0 * values['order_accuracy']:.2f}%), "
                    f"last task {values['correct_last_tasks']}/{total} "
                    f"({100.0 * values['last_task_accuracy']:.2f}%)"
                )

    if weight_matching_results is not None:
        print()
        print("-" * 70)
        print("WEIGHT MATCHING")
        print("-" * 70)

        for metric, values in weight_matching_results["summary"]["metrics"].items():
            print(
                f"{metric:>28}: "
                f"{values['correct_predictions']}/{total} "
                f"({100.0 * values['prediction_accuracy']:.2f}%)"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = result_root / "combined" / representation_mode

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
