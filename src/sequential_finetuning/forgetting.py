from typing import Dict, List


def calculate_forgetting(
    stage_results: List[Dict],
) -> Dict[str, float]:
    """
    Calculate forgetting for each task.

    For a task T:

        forgetting(T) =
            accuracy immediately after learning T
            -
            final accuracy on T

    Positive values indicate forgetting.
    Negative values indicate that final performance
    improved relative to the task's first evaluation.
    """

    learned_accuracy = {}
    final_accuracy = {}

    # ---------------------------------------------------------
    # Accuracy immediately after each task was learned
    # ---------------------------------------------------------

    for stage in stage_results:

        learned_task = stage["learned_task"]

        for evaluation in stage["evaluations"]:

            task = evaluation["task"]

            if task == learned_task:
                learned_accuracy[task] = evaluation["accuracy"]

    # ---------------------------------------------------------
    # Accuracy at the final stage
    # ---------------------------------------------------------

    if not stage_results:
        return {}

    final_stage = stage_results[-1]

    for evaluation in final_stage["evaluations"]:

        task = evaluation["task"]

        final_accuracy[task] = evaluation["accuracy"]

    # ---------------------------------------------------------
    # Calculate forgetting
    # ---------------------------------------------------------

    forgetting = {}

    for task in learned_accuracy:

        if task not in final_accuracy:
            continue

        forgetting[task] = learned_accuracy[task] - final_accuracy[task]

    return forgetting
