# Figure Guide

`scripts/plot_combined_results.py` creates paired 300 DPI PNG and PDF figures from the completed combined analysis. Sequential-order labels are generated from the recorded task sequence, for example `A -> B -> C`. A black outline always marks the task that was actually learned last.

## Last-Task Prediction Accuracy

`last_task_prediction_accuracy.png/.pdf` compares four explicit fingerprinting rules: full-model weight distance, backbone weight distance, CKA, and feature drift. The x-axis is method; the y-axis is the percentage of sequential orders for which its predicted task matches the actual last task. The current experiment evaluates six orders.

The dashed horizontal reference is the **Always-A baseline**. It is computed from the results: because A, B, and C each occur last twice, always predicting A yields $2/6 = 33.3\%$. It is not a majority-class baseline. Accuracy above it suggests potentially useful last-task information, but six orders are very small-sample evidence. Stronger evidence requires both higher accuracy and non-biased predictions.

## Prediction Matrix

`prediction_matrix.png/.pdf` has one row per sequential order. The first column is the actual last task and the remaining columns contain each method's predicted task. Green cells are correct predictions; red cells are incorrect.

This figure exposes errors that a single accuracy hides. In particular, a column repeatedly containing A can achieve 33.3% in the current balanced experiment without recovering the learned order. Evidence for fingerprinting is correct predictions across different actual last tasks, rather than repeated selection of one task.

## Weight-Distance Scores

`weight_distance_scores.png/.pdf` compares each final sequential model with every `Single-<task>` reference. The x-axis is sequential order and the y-axis is weight distance. Colors identify reference tasks; black outlines mark the reference matching the actual last task. The upper panel uses the complete model and the lower panel uses the backbone only.

Lower distance means greater parameter similarity, and the prediction rule selects the smallest distance. The numerical bars matter because they show the separation or ambiguity behind a label: a correct prediction with only a tiny margin is weaker evidence than repeated, clear minima at the actual last task. Distance alone does not establish a causal memory mechanism.

## CKA Similarity

`cka_similarity.png/.pdf` presents the final sequential model's representation similarity to each single-task reference, separately for each order. The x-axis is reference task and the y-axis is CKA; bar colors identify reference tasks and black outlines identify the actual last task.

Higher CKA means more similar representations, so the prediction rule chooses the highest bar. Consistently highest CKA for the actual last task supports the fingerprinting hypothesis; highest bars concentrated on another task or weak score separation argue against it.

## Feature Drift

`feature_drift.png/.pdf` also compares representations to single-task references. Its x-axis is reference task and its y-axis is feature drift. Colors and black outlines have the same meaning as the CKA figure.

Lower drift means more similar representations, so the prediction rule chooses the lowest bar. CKA and feature drift are related representation comparisons but quantify different properties; agreement provides more confidence than assuming one score is interchangeable with the other. A low-drift actual-last-task reference across orders supports the hypothesis, while biased or weakly separated minima do not.

## Method Agreement

`method_agreement.png/.pdf` has one row per order and one column per fingerprinting method. Each cell contains the predicted task and is colored by that task; the y-axis label separately states the actual last task.

Agreement among methods that is also correct across varied orders is evidence of a consistent fingerprint. Disagreement can reveal method-specific sensitivity or weak evidence. Agreement on the same wrong task is systematic bias, not support for fingerprinting.

## Forgetting

`forgetting_by_order.png/.pdf` gives one panel per order. The x-axis is task and the y-axis is

$$\operatorname{forgetting}(t) = \operatorname{accuracy\ after\ learning\ }t - \operatorname{final\ accuracy\ on\ }t.$$

Positive values indicate forgetting, zero means no change, and negative values mean final accuracy improved. Colors identify tasks and the black outline marks the actual last task. This is supporting evidence about sequential-learning dynamics, not a direct last-task classifier. For example, lower forgetting of the final task can help explain a representation pattern, but it does not itself predict recency.

## Loss Barriers

`loss_barrier_curves.png/.pdf` contains one panel per order. It evaluates the **actual last task** while interpolating weights between the final sequential model and each single-task reference. The x-axis is interpolation alpha, where $\alpha=0$ is the reference model and $\alpha=1$ is the final sequential model; the y-axis is loss. Colored curves identify `Single-<task>` references, and dashed vertical lines mark both endpoints.

`loss_barrier_height.png/.pdf` and `loss_barrier_area.png/.pdf` summarize the same actual-last-task slice. Rows are sequential orders, columns are reference models, and black cell outlines identify the reference for the actual last task. Barrier height is the maximum loss above the endpoint reference level; barrier area summarizes excess loss over the interpolation path. Lower or otherwise distinctive barriers may show geometric relationships to a reference, but neither metric is a last-task classifier here because no prediction rule has been defined or validated for it.

## Jacobian Sensitivity

`jacobian_sensitivity_by_order.png/.pdf` shows mean Jacobian sensitivity of each sequential model on each task. The x-axis is task and the y-axis is mean sensitivity; colors identify tasks and black outlines mark the actual last task. Jacobian sensitivity measures how much model output changes for changes in the input. Elevated sensitivity for the last task may be diagnostically interesting, but higher sensitivity does not automatically mean more recently learned.

`jacobian_vs_single_references.png/.pdf` keeps task identity explicit. Each panel evaluates one task; bars give the sequential model's mean sensitivity by order and colored lines show each single-task reference. It helps assess whether sequential sensitivity resembles a particular reference, without collapsing tasks into a single average.

`jacobian_channel_sensitivity.png/.pdf` is a diagnostic heatmap. Rows are `order / task`, columns are R, G, and B input channels, and values are channel sensitivities. It may reveal channel-specific effects, but is not a direct recency classifier. All Jacobian figures are supporting evidence unless a separate, validated prediction rule is introduced.

## Research Summary

`research_summary.png/.pdf` is the compact thesis-oriented overview. Its chart reports accuracy, the dynamically computed Always-A baseline, and correct predictions out of the total number of orders. The table reports each method's prediction distribution over tasks.

The distribution is essential: 33.3% can result from always predicting A in this balanced six-order experiment, so accuracy alone does not show recovery of training order. A useful method needs both correct predictions and a distribution that tracks the actual last tasks.

## Interpreting the Suite

**Stronger evidence** for the last-task fingerprinting hypothesis would be accuracy clearly above the Always-A baseline, predictions spread according to actual last tasks rather than concentrated on one task, numerical similarity scores consistently favoring the actual last task, and repetition of that pattern across orders.

**Weak evidence** includes 33.3% accuracy, prediction concentration on one task, small score differences between references, or strong disagreement among methods. Forgetting, loss barriers, and Jacobian sensitivity should explain model behavior rather than be treated automatically as classifiers. With only six sequential orders, conclusions should remain cautious and be tested on additional runs, seeds, tasks, or task orders.
