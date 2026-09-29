import json
from pathlib import Path
from typing import Dict, List


def load_json(path: Path) -> Dict:
    """Load a JSON result file."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def index_orders(results: Dict) -> Dict[str, Dict]:
    """
    Convert an analysis result's order list into a lookup dictionary.

    Example:
        ["A", "B", "C"] -> "A_B_C"
    """
    return {"_".join(entry["order"]): entry for entry in results["orders"]}


def optional_order_entry(
    results: Dict | None,
    order_name: str,
) -> Dict | None:
    """Look up one order in an optional analysis result file."""

    if results is None:
        return None

    if isinstance(results, dict) and "orders" in results:
        return index_orders(results).get(order_name)

    return results.get(order_name)


def combine_order_results(
    order: List[str],
    sequential_results: Dict,
    weight_results: Dict,
    representation_results: Dict,
    loss_barrier_results: Dict | None = None,
    jacobian_results: Dict | None = None,
    fisher_results: Dict | None = None,
    weight_matching_results: Dict | None = None,
) -> Dict:
    """Combine all existing analyses for one task order."""

    order_name = "_".join(order)

    sequential_by_order = index_orders(sequential_results)
    weight_by_order = index_orders(weight_results)
    representation_by_order = index_orders(representation_results)

    if order_name not in sequential_by_order:
        raise KeyError(f"Order {order_name} not found in sequential results.")

    if order_name not in weight_by_order:
        raise KeyError(f"Order {order_name} not found in weight-distance results.")

    if order_name not in representation_by_order:
        raise KeyError(f"Order {order_name} not found in representation results.")

    sequential = sequential_by_order[order_name]
    weight = weight_by_order[order_name]
    representation = representation_by_order[order_name]

    loss_barrier_data = optional_order_entry(loss_barrier_results, order_name)

    jacobian_data = optional_order_entry(jacobian_results, order_name)

    fisher_data = optional_order_entry(fisher_results, order_name)

    weight_matching_data = optional_order_entry(weight_matching_results, order_name)

    result = {
        "order": list(order),
        "order_name": order_name,
        "actual_last_task": order[-1],
        "forgetting": sequential["forgetting"],
        "weight_distance": {
            "full_model": {
                "distances": weight["full_model"]["distances"],
                "predicted_last_task": weight["full_model"]["predicted_last_task"],
                "correct": weight["full_model"]["correct"],
            },
            "backbone": {
                "distances": weight["backbone"]["distances"],
                "predicted_last_task": weight["backbone"]["predicted_last_task"],
                "correct": weight["backbone"]["correct"],
            },
        },
        "representation": {
            "evaluation_set": representation_results.get("evaluation_set", {}).get(
                "mode", "probe"
            ),
            "cka": {
                task: representation["comparisons"][task]["cka"]
                for task in representation["comparisons"]
            },
            "feature_drift": {
                task: representation["comparisons"][task]["feature_drift"]
                for task in representation["comparisons"]
            },
            "predicted_last_task": representation["prediction"]["cka"],
            "cka_correct": representation["correct"]["cka"],
            "drift_predicted_last_task": representation["prediction"]["feature_drift"],
            "drift_correct": representation["correct"]["feature_drift"],
        },
        "checkpoints": {
            "sequential": sequential.get("checkpoint"),
            "weight_distance": weight.get("checkpoint"),
            "representation": representation.get("checkpoint"),
        },
    }

    if loss_barrier_data is not None:
        result["loss_barrier"] = loss_barrier_data

    if jacobian_data is not None:
        result["jacobian"] = jacobian_data

    if fisher_data is not None:
        result["fisher"] = {
            "scores": fisher_data["scores"],
            "predictions": fisher_data["predictions"],
        }

    if weight_matching_data is not None:
        result["weight_matching"] = {
            "predictions": weight_matching_data["predictions"],
            "comparisons": {
                task: {
                    "l2": comparison["l2"],
                    "cosine": comparison["cosine"],
                    "verification": comparison["verification"],
                }
                for task, comparison in weight_matching_data["comparisons"].items()
            },
        }

    return result


def combine_all_results(
    orders: List[List[str]],
    sequential_results: Dict,
    weight_results: Dict,
    representation_results: Dict,
    loss_barrier_results: Dict | None = None,
    jacobian_results: Dict | None = None,
    fisher_results: Dict | None = None,
    weight_matching_results: Dict | None = None,
) -> Dict:
    """Combine all configured sequential orders."""

    combined_orders = []

    for order in orders:
        combined_orders.append(
            combine_order_results(
                order=order,
                sequential_results=sequential_results,
                weight_results=weight_results,
                representation_results=representation_results,
                loss_barrier_results=loss_barrier_results,
                jacobian_results=jacobian_results,
                fisher_results=fisher_results,
                weight_matching_results=weight_matching_results,
            )
        )

    return {
        "analysis": "combined",
        "representation_evaluation_set": representation_results.get(
            "evaluation_set", {}
        ),
        "orders": combined_orders,
    }
