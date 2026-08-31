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


def combine_order_results(
    order: List[str],
    sequential_results: Dict,
    weight_results: Dict,
    representation_results: Dict,
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

    return {
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


def combine_all_results(
    orders: List[List[str]],
    sequential_results: Dict,
    weight_results: Dict,
    representation_results: Dict,
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
            )
        )

    return {
        "analysis": "combined",
        "orders": combined_orders,
    }
