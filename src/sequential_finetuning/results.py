import json
from pathlib import Path
from typing import Dict, List


def save_json(data: dict, path: str | Path) -> None:
    """
    Save a dictionary as formatted JSON.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


def calculate_forgetting(
    stage_results: List[Dict],
) -> Dict[str, float]:
    """
    Calculate forgetting for every task that was learned
    before the final stage.

    Forgetting(task) =
        accuracy immediately after learning task
        -
        final accuracy on task
    """

    learned_accuracy = {}

    final_accuracy = {}

    # --------------------------------------------------------
    # Find accuracy immediately after learning each task
    # --------------------------------------------------------

    for stage in stage_results:

        learned_task = stage["learned_task"]

        for evaluation in stage["evaluations"]:

            task = evaluation["task"]

            accuracy = evaluation["test_accuracy"]

            if task == learned_task:
                learned_accuracy[task] = accuracy

    # --------------------------------------------------------
    # Find final accuracy for every task
    # --------------------------------------------------------

    final_stage = stage_results[-1]

    for evaluation in final_stage["evaluations"]:

        task = evaluation["task"]

        final_accuracy[task] = evaluation["test_accuracy"]

    # --------------------------------------------------------
    # Calculate forgetting
    # --------------------------------------------------------

    forgetting = {}

    for task in learned_accuracy:

        forgetting[task] = learned_accuracy[task] - final_accuracy[task]

    return forgetting
