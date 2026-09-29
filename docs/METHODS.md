# Sequential Fine-Tuning Order Fingerprints: Scientific Methods

## 1. Research Question

**Primary Question:**
> Does the final checkpoint of a sequentially fine-tuned model retain a measurable geometric fingerprint that reveals which task was learned most recently?

**Specific Formulation:**

Given six task orders (all permutations of A, B, C):
- A → B → C
- A → C → B
- B → A → C
- B → C → A
- C → A → B
- C → B → A

Can we predict the last task (A, B, or C) by comparing the final sequential checkpoint against three single-task reference models?

---

## 2. Sequential Fine-Tuning Setup

### Training Process

1. **Single-task reference models:** Train three independent models, each on a single task (A, B, or C)
   - Used as reference models for comparison
   - Represent the "pure" learned representation for each task

2. **Sequential models:** For each of the six task orders, train sequentially:
   - Single shared backbone (ResNet18)
   - Multiple task-specific heads (one per task)
   - Continual learning without task boundaries
   - No explicit mechanism to prevent forgetting

3. **Checkpoint hierarchy:**
   - Save intermediate checkpoints after each task
   - Save final checkpoint after all three tasks
   - Enables analysis of both intermediate and final states

---

## 3. Tasks

Three 5-class CIFAR-100 superclass-based classification tasks:

**Task A:**
- beaver, dolphin, otter, seal, whale
- Aquatic animals

**Task B:**
- clock, keyboard, lamp, telephone, television
- Electronic/office devices

**Task C:**
- bicycle, bus, motorcycle, pickup_truck, train
- Vehicles

---

## 4. Single-Task Reference Models

### Purpose

Reference models establish the "ground truth" learned representation for each task in isolation:
- Trained independently with no task interference
- No forgetting of earlier tasks (there are none)
- Represent optimal task-specific representation

### Training

Each reference model is trained with:
- Full ResNet18 backbone
- Single task-specific classification head (5 classes)
- Standard training hyperparameters (SGD, momentum, weight decay)
- Fixed random seed for reproducibility

### Use in Comparisons

All three comparison methods (weight distance, CKA, feature drift) use these reference models as baselines:
- Sequential model is compared against each reference
- Predictions are made based on which reference is most similar

---

## 5. Sequential Models

### Architecture

- **Backbone:** Shared ResNet18 across all tasks
- **Heads:** Multiple task-specific classification heads
  - One 5-class head per task learned so far
  - Task-specific outputs selected by task identifier

### Training Protocol

For each sequential order (e.g., A → B → C):

1. **Stage 1:** Train backbone and A-head on task A
2. **Stage 2:** Freeze A-head, extend backbone with B-head, train on task B
3. **Stage 3:** Freeze A-head and B-head, extend backbone with C-head, train on task C

### Continual Learning Characteristics

- **No explicit regularization:** No EWC, SI, or replay
- **Full learning rate:** Each stage uses standard learning rate
- **Single backbone:** Shared representation space for all tasks
- **Catastrophic forgetting expected:** Earlier tasks may degrade

---

## 6. Checkpoint Convention

### Naming

Sequential checkpoints use concatenated task identifiers:
```
checkpoints/sequential/<ORDER>/<CHECKPOINT>.pt
```

Examples:
- `A_B_C/A.pt` → After training A in order A→B→C
- `A_B_C/A_B.pt` → After training A and B in order A→B→C
- `A_B_C/A_B_C.pt` → Final checkpoint after A, B, C

### Determination of Last Task

The last task is always `order[-1]`:
```python
order = ["A", "B", "C"]
last_task = order[-1]  # "C"
```

### Metadata

Each checkpoint contains:
- Model state dictionary
- Task history (e.g., ["A", "B", "C"])
- Current task (e.g., "C")
- Order as list
- Epoch and metrics
- Configuration used

---

## 7. Fixed Representation Probe

### Motivation

To enable fair comparison of representations across models, we need:
1. **Deterministic:** Same images across all model evaluations
2. **Comprehensive:** Representative samples from all tasks
3. **Fixed-size:** Consistent for statistical comparisons
4. **Reproducible:** Seeded random selection

### Specification

**Composition:**
- All 15 classes from tasks A, B, and C combined
- 20 images per class (randomly selected from CIFAR-100 test set)
- 300 total images
- Seed: 42 (fixed)

**Creation:**
Generated once by `scripts/create_probe.py` using:
```python
classes = []
for task_classes in config["tasks"].values():
    classes.extend(task_classes)

rng = random.Random(42)
for class_name in classes:
    selected = rng.sample(class_indices, 20)
    probe_indices[class_name] = selected
```

**Reuse:**
All representation analyses (single-task, sequential, test) use the same `probe_set.json`:
- Ensures comparability
- Prevents p-hacking through probe selection
- Enables deterministic results

### 7.1 Representation Evaluation-Set Modes

`representation.probe.enabled` in `configs/experiment.yaml` selects the CKA/feature-drift evaluation set. `true` (default) preserves the probe behavior described above unchanged. `false` uses every CIFAR-100 test image belonging to the union of the configured task classes, currently 1,500 images across 15 classes, with no random subsampling and no training images. The complete set is constructed once, iterated in deterministic dataset order, and reused for the sequential model and all single-task references. Feature extraction remains batched, moving each batch to CPU before continuing, so the full test set does not need to reside on GPU at once.

Probe mode is computationally cheaper.

---

## 8. Weight-Space Comparison

### Overview

Direct parameter-space distance between sequential and single-task reference models.

### Method 8.1: Full-Model L2 Distance

**Definition:**
```
D_full = ||θ_sequential - θ_reference||_2
```

where θ includes all parameters:
- Shared backbone (ResNet18 features)
- All task-specific heads

**Interpretation:**
- Measures complete parameter divergence
- Lower distance suggests more similar learned parameters
- Affected by both backbone and head parameters

**Advantages:**
- Captures entire parameter state
- Task-specific heads encode recent learning

**Disadvantages:**
- Task-specific heads from reference models are unused
- May not reflect true representation similarity

### Method 8.2: Backbone-Only L2 Distance

**Definition:**
```
D_backbone = ||θ_backbone_sequential - θ_backbone_reference||_2
```

where θ_backbone contains only ResNet18 shared parameters.

**Interpretation:**
- Measures only shared representation parameter distance
- Task-specific heads excluded
- Focuses on the learned feature space

**Advantages:**
- Isolates backbone representation from head effects
- More directly comparable (same head architectures)
- Reflects shared representation similarity

**Disadvantages:**
- Ignores task-specific adaptations in final checkpoint
- Cannot capture task-head specific parameter changes

### Prediction Rule

**Minimum distance predicts last task:**
```python
prediction = argmin_task(D[task])
```

The task corresponding to the smallest distance is predicted as the last task.

---

## 9. Representation-Space Comparison

### Overview

Indirect comparison through learned feature representations extracted from intermediate layers.

### Method 9.1: Feature Extraction

**Process:**
1. Load model (reference or sequential)
2. Set to evaluation mode
3. For each batch in probe images:
   - Forward pass to backbone
   - Extract intermediate features (penultimate layer)
   - Concatenate across all probe images

**Output:**
Matrix of shape `[300, 512]` where:
- 300 = probe size
- 512 = ResNet18 penultimate layer dimension

### Method 9.2: Linear CKA (Centered Kernel Alignment)

**Definition:**
CKA is a representation-similarity metric invariant to orthogonal transformations:

```
CKA(X, Y) = 
    ||X^T Y||_F^2 / 
    (||X^T X||_F * ||Y^T Y||_F)^0.5
```

where X and Y are centered feature matrices.

**Interpretation:**
- CKA ∈ [0, 1]
- High CKA (→ 1): Representations are structurally similar
- Low CKA (→ 0): Representations differ substantially
- Invariant to orthogonal transformations (unlike Euclidean distance)

**Advantages:**
- Captures structural alignment without rotation
- Two networks can have identical CKA with different parameters
- Robust to scaling differences

**Disadvantages:**
- Does not measure feature magnitude (only structure)
- Can be high even if individual features drift

### Method 9.3: Feature Drift

**Definition:**
Sample-wise L2 distance between normalized feature vectors:

```
drift(X, Y) = mean(||x_normalized - y_normalized||_2)
```

where:
- `x_normalized = x / ||x||_2` (L2-normalized per sample)
- Computed over all probe samples

**Interpretation:**
- drift ∈ [0, 2] (range for L2-normalized vectors)
- Low drift: Sequential model's features are close to reference
- High drift: Features have moved substantially

**Advantages:**
- Sample-level granularity
- Captures magnitude changes
- Complements CKA (structural vs. magnitude)

**Disadvantages:**
- Affected by magnitude scaling
- Does not capture covariance changes

### Comparison of CKA and Feature Drift

Both methods answer different questions:

**CKA:**
- "Is the representation structure similar?"
- High CKA → last task left structural fingerprint

**Feature Drift:**
- "Have the specific feature vectors changed?"
- Low drift → last task left feature-level fingerprint

A model might have:
- High CKA but high drift (similar structure, magnitude changed)
- Low CKA but low drift (different structure, but close features)

### Prediction Rule

**Maximum CKA predicts last task:**
```python
prediction_cka = argmax_task(CKA[task])
```

**Minimum feature drift predicts last task:**
```python
prediction_drift = argmin_task(drift[task])
```

---

## 10. Last-Task Prediction Rule

### Summary of Prediction Methods

| Method | Comparison | Prediction Rule | Intuition |
|--------|-----------|-----------------|-----------|
| Full-model L2 | Parameter space | Minimum distance | Closest in weight space |
| Backbone L2 | Backbone parameters | Minimum distance | Most similar backbone |
| CKA | Feature space structure | Maximum similarity | Most similar representation |
| Feature drift | Feature space magnitude | Minimum drift | Smallest feature movement |
| Matched L2 (Section 15) | Parameter space modulo permutation | Minimum distance | Closest after removing neuron reordering |
| Task-vector cosine (Section 15) | Update direction | Maximum similarity | Moved in the most similar direction |
| Fisher trace (Section 14) | Curvature of the final model | Sorted ranking, both directions | Order fingerprint without reference models |

The Fisher ranking additionally yields a guess for the complete training order; all other methods only predict the last task.

### Correctness Evaluation

For each sequential order:
1. Determine true last task: `actual_last = order[-1]`
2. Apply each prediction method
3. Check: `predicted == actual`
4. Compute accuracy across all six orders

**Expected accuracy if random:** 33.33% (1 out of 3 tasks correct)

---

## 11. Forgetting

### Definition

**Forgetting** measures the loss of performance on an earlier task after learning additional tasks:

```
forgetting(T) = 
    accuracy(T) immediately after learning T
    -
    accuracy(T) at the end of training
```

### Interpretation

- **Positive value (> 0):** Task T degraded → forgetting occurred
- **Zero (= 0):** Task T maintained performance
- **Negative value (< 0):** Task T improved → no forgetting, possible benefit

### Example

For order A → B → C:
- Accuracy on A immediately after A: 0.45
- Accuracy on A at the end (after B and C): 0.446
- Forgetting(A) = 0.45 - 0.446 = 0.004 (minimal forgetting)

### Relationship to Last-Task Fingerprint Hypothesis

If the last task leaves a fingerprint, we might expect:
- Earlier tasks have higher forgetting
- Last task has minimal forgetting
- Representation is most similar to last task

---

## 12. Loss-Barrier Curves

### Motivation

Loss-barrier analysis measures the "smoothness" of the optimization landscape between two models by interpolating in weight space and evaluating loss at intermediate points.

A high barrier indicates that sequential and single-task models found different local minima.
A low barrier indicates they found similar solutions in the loss landscape.

### Method

**Linear Weight-Space Interpolation:**

```
θ(α) = (1 - α) * θ_reference + α * θ_sequential
```

where α ranges from 0 (reference weights) to 1 (sequential weights).

**Procedure:**

1. Load sequential model weights
2. Load single-task reference model weights
3. For each interpolation point α ∈ {0, 0.1, 0.2, ..., 1.0}:
   - Create interpolated model: θ(α)
   - Set model weights to θ(α)
   - Evaluate loss on task's test set
   - Record loss value
4. Compute barrier height = max(losses) - min(losses)
5. Compute barrier area using trapezoidal integration

### Interpretation

**Low Barrier (< 0.1):**
- Smooth interpolation between models
- Both models found similar solutions
- Suggests compatible learned representations

**High Barrier (> 0.5):**
- Loss spike between models
- Different local minima
- Suggests divergent learning trajectories

**Barrier Shape:**
- Symmetric barrier: Equally distant from both minima
- Asymmetric barrier: One model closer to global optimum
- Multiple peaks: Complex loss landscape structure

### Task-Specific Analysis

Barriers are computed on each task's test set:
- **On last task:** Expected to have low barrier (most recent training)
- **On earlier tasks:** May have higher barrier due to forgetting
- **Cross-task barriers:** Measure representation divergence for other tasks

---

## 13. Jacobian Sensitivity Analysis

### Motivation

Input-output Jacobian sensitivity reveals how sensitive model predictions are to input perturbations on task probe images.

High sensitivity indicates that model outputs are strongly dependent on input features.
Low sensitivity indicates robust or invariant predictions.

Task-specific sensitivity patterns can reveal which inputs most influence predictions on each task.

### Method

**Jacobian Computation:**

For each probe image, compute:
```
J[i, j, k, l] = ∂(output[i, j]) / ∂(input[i, k, l])
```

where:
- i = batch index
- j = class index (softmax output)
- k, l = spatial coordinates (H=32, W=32)

For computational efficiency, compute Jacobian only for predicted class:
```
J[i, k, l] = ∂(predicted_logit[i]) / ∂(input[i, k, l])
```

**Procedure:**

1. Load model (reference or sequential)
2. Set to evaluation mode
3. For each batch of probe images:
   - Enable gradient computation on inputs
   - Forward pass to obtain predictions
   - Backward pass to compute input gradients
   - Compute Jacobian norm: ||J|| = sqrt(sum(J^2))
4. Aggregate across all probe images

### Sensitivity Metrics

**Mean Sensitivity:**
Average Jacobian norm across all probe images.
High value indicates strong dependence on inputs.

**Spatial Sensitivity Map:**
Average absolute gradient magnitude at each spatial location.
Shows which regions (center, edges, corners) most influence predictions.

**Channel Sensitivity:**
Average gradient magnitude per color channel (R, G, B).
Reveals which color channel carries most predictive information.

### Interpretation

**High Sensitivity (mean > 1.0):**
- Model predictions strongly influenced by inputs
- High reliance on visual features
- May indicate overfitting to specific image regions

**Low Sensitivity (mean < 0.1):**
- Robust predictions regardless of input perturbations
- Features learned to be invariant
- May indicate insufficient training or simplistic solution

**Task Differences:**
- Some tasks may require higher sensitivity to specific features
- Sequential models may show different sensitivity than single-task models
- Sensitivity can reveal whether sequential training altered feature importance

---

## 14. Fisher Information

### Motivation

All methods above need reference models. The Fisher information matrix is a property of the **final checkpoint alone**: it measures how sharply the model's own predictive distribution reacts to parameter changes for the data of a given task.

If sequential fine-tuning leaves an order fingerprint in the curvature of the solution, the three per-task Fisher traces `F_A`, `F_B`, `F_C` should be ordered consistently with the training order.

### Estimator

The **true** (model-sampled) Fisher is used, not the empirical one. For a task `T` with samples `x_1 ... x_N`:

```
F_T = (1 / N) * sum_n sum_p ( d/dtheta_p  -log p( y_n_hat | x_n ) )^2

with y_n_hat ~ p( . | x_n )
```

The dataset ground-truth label is deliberately **not** used: the label is drawn from the model's own predicted class probabilities. This makes `F_T` the trace of the Fisher information matrix rather than a measure of fit quality.

### Procedure

For each of the three tasks, and one sample at a time:

1. Run the sample through the final checkpoint (the task's own head is used).
2. Draw a label at random from the predicted class probabilities.
3. Compute the cross-entropy loss for that sampled label and back-propagate.
4. Square every gradient entry and accumulate.
5. Clear the gradients. **No optimizer step is taken**, the weights never change.

Averaging over the samples and summing over all weights yields one scalar per task. The analysis additionally reports the same quantity restricted to the shared backbone (`fisher_backbone`) and to the task head (`fisher_head`); the heads of the other two tasks receive no gradient and contribute zero.

### Sample Selection

`analysis.fisher` in `configs/experiment.yaml` controls the estimate:

- `num_samples` (default 1000) samples per task,
- `split` (default `train`) selects the CIFAR-100 split,
- `seed` controls both the subset selection and the label sampling.

Deterministic (test-set) preprocessing is used even for the train split, and the identical sample subset is reused for every checkpoint, so the traces are comparable across orders.

### Prediction Rule

Sorting `F_A`, `F_B`, `F_C` gives a guessed training order. Whether a high Fisher trace means "learned recently" or "learned long ago" is not known a priori, so both reading directions are computed and evaluated:

| Hypothesis | Guessed order | Guessed last task |
|------------|---------------|-------------------|
| `high_is_recent` | ascending in `F` | task with the largest `F` |
| `low_is_recent` | descending in `F` | task with the smallest `F` |

For each direction the full guessed order and the guessed last task are compared against the ground truth. Random chance is 1/6 for the full order and 1/3 for the last task.

---

## 15. Weight Matching (Git Re-Basin)

### Motivation

Neural networks have permutation symmetries: the channels of a convolution can be reordered, and with the matching reordering of the following layer the network computes exactly the same function. Two functionally similar models can therefore be far apart in L2 distance simply because their neurons are numbered differently, which makes the plain weight distance of Section 8 misleading.

Weight matching removes this symmetry before measuring distance.

### Method

The implementation follows the weight-matching algorithm of Ainsworth, Hayase and Srinivasa, *Git Re-Basin: Merging Models modulo Permutation Symmetries* (ICLR 2023).

**Permutation specification.** Every parameter axis is assigned to a permutation group. For the ResNet18 backbone used here:

- `P_stem` — output channels of `conv1`/`bn1`. Because the blocks of `layer1` use identity shortcuts, this is also the residual stream of `layer1`.
- `P_layer2`, `P_layer3`, `P_layer4` — residual streams of the remaining stages. The downsample convolution writes into the same stream and therefore shares the permutation.
- `P_layerX_Y_inner` — output of the first convolution inside block `Y`, free because it is consumed only by the second convolution.

Batch-norm weights, biases and running statistics follow the channel permutation of their layer; the running statistics are permuted but excluded from the matching cost. The 5 output units of each task head are class labels and are never permuted. The head input dimension is tied to `P_layer4`.

**Optimization.** Holding all other groups fixed, the optimal permutation of one group maximizes

```
sum_i  < w_reference[i] , w_target[perm(i)] >
```

summed over all parameter axes tied to that group. This is a linear assignment problem, solved exactly with the Hungarian algorithm. The groups are visited in random order and the sweep is repeated until no group changes any more (coordinate descent). The procedure is started from the identity permutation.

**Verification.** After alignment the permuted checkpoint is evaluated on a fixed batch of test images for all three heads and compared with the original checkpoint. The maximum absolute logit difference must stay below `analysis.weight_matching.verification_tolerance`; otherwise the analysis aborts. This proves that the alignment only relabeled neurons and did not change the function.

### Metrics

For every sequential final checkpoint and every single-task reference:

1. **L2 before matching** — the distance of Section 8, repeated here for direct comparison.
2. **L2 after matching** — the distance between the aligned checkpoint and the reference, full model and backbone only.
3. **Task-vector cosine similarity**

```
cos( theta_final - theta_init ,  theta_task - theta_init )
```

   where `theta_init` is the shared initialization stored in
   `checkpoints/verified_initialization/base_model.pt`. Unlike L2, this is not
   affected by how far training moved the weights in total; it only compares
   the direction of movement. It is computed for the whole network, for the
   backbone only, and per layer.

   Note that a single-task reference never updates the heads of the other two
   tasks: those parameters still hold their initialization values, so their
   direction is exactly zero. Per-layer cosines are reported as `null` for
   those layers, and in the full-model cosine they enlarge the norm of the
   sequential task vector without contributing to the inner product. The
   backbone cosine is free of this asymmetry.

### Prediction Rule

- Minimum matched L2 distance predicts the last task.
- Maximum cosine similarity predicts the last task, because the most recent task should point in the most similar direction.

Both are reported with and without alignment so the effect of weight matching is visible.

### Expected Behaviour in This Project

All checkpoints in this project descend from one shared initialization and are fine-tuned for a limited number of epochs, so they are expected to stay in the same basin. Weight matching then returns the identity permutation and leaves the distances unchanged. This is a meaningful negative control: it shows that the plain L2 results of Section 8 are *not* an artifact of neuron reordering. Weight matching becomes essential as soon as checkpoints with different initializations are compared.

---

## 16. Combined Analysis

### Purpose

Synthesize the results of the independent analyses into a single interpretable summary.

### Inputs

1. **Sequential evaluation:** Forgetting for each task
2. **Weight-distance analysis:** Predictions from full-model and backbone L2
3. **Representation analysis:** Predictions from CKA and feature drift
4. **Fisher analysis (when enabled):** Per-task traces and both ranking directions
5. **Weight-matching analysis (when enabled):** Matched distances and cosine similarities
6. **Loss-barrier and Jacobian analyses (when enabled)**

### Output

Single JSON file combining all results:
```json
{
  "order": ["A", "B", "C"],
  "actual_last_task": "C",
  "forgetting": {"A": 0.004, "B": 0.040, "C": 0.000},
  "weight_distance": {
    "full_model": {"prediction": "A", "correct": false},
    "backbone": {"prediction": "A", "correct": false}
  },
  "representation": {
    "cka": {"prediction": "A", "correct": false},
    "feature_drift": {"prediction": "A", "correct": false}
  }
}
```

### No Recalculation

The combined analysis **does not recalculate** any metrics. It only:
1. Reads existing result JSON files
2. Indexes them by order
3. Merges them into unified structure
4. Computes summary statistics (e.g., prediction accuracy across all methods)

---

## 17. Reproducibility

### Two Independent Seeds

1. **Global Experiment Seed (`seed: 42`)**
   - Controls training initialization
   - Controls weight initialization
   - Controls data shuffling
   - Ensures deterministic training given the same hardware

2. **Representation Probe Seed (`representation.probe.seed: 42`)**
   - Controls which CIFAR-100 test images are selected
   - Independent from training seed
   - Ensures same probe used across all model evaluations
   - Allows changing one seed without changing the other

### Why Two Seeds?

- **Training seed:** For reproducible model training
- **Probe seed:** For reproducible representation analysis independent of training

This separation allows:
- Reusing a probe across different training runs
- Changing training seed without recomputing probe
- Systematic variation of probe while fixing training

### Hardware Considerations

Results may vary slightly across:
- Different GPUs (different numerical precision)
- Different PyTorch/CUDA versions
- Different computational order (floating-point non-associativity)

The seeds ensure **deterministic execution on the same hardware**.

---

## 18. Summary of Hypotheses and Methods

### H1: Last-Task Fingerprint

**Hypothesis:** Sequential fine-tuning leaves a measurable fingerprint of the last task.

**Test:** Every method should predict the last task better than random (>33%).

**Evidence from the current experiment** (20 epochs per task, all-test evaluation set, six orders):
- CKA: 6/6, feature drift: 5/6 — clearly above chance
- Full-model and backbone L2: 0/6, also after weight matching
- Fisher trace with `low_is_recent`: 4/6

### H2: Representation > Weight Space

**Hypothesis:** Representation-level similarity (CKA, drift) captures task-order effects better than parameter-space distance.

**Test:** Compare the accuracy of weight distance against the representation methods.

**Current status:** Supported. Representation methods reach 83-100%, while parameter-space distance reaches 0% and systematically points at the first task instead of the last one. Weight matching shows that this is not caused by permutation symmetry.

### H3: Forgetting Correlation

**Hypothesis:** Last task has minimal forgetting, earlier tasks have higher forgetting.

**Test:** Compare forgetting magnitudes across tasks and orders.

**Current observation:** Variable forgetting patterns, no clear last-task advantage

### H4: Curvature Fingerprint

**Hypothesis:** The Fisher information trace of the final checkpoint depends on when a task was learned.

**Test:** Rank `F_A`, `F_B`, `F_C` and evaluate both reading directions.

**Current observation:** The `low_is_recent` direction predicts the last task in 4/6 runs and the complete order in 1/6; the `high_is_recent` direction is never correct. In this setup the most recently learned task therefore has the *lowest* Fisher trace.

---

## 19. Interpretation and Limitations

### Current Results

With 20 epochs per task and the all-test evaluation set: CKA 6/6, feature drift 5/6, Fisher (`low_is_recent`) 4/6, full-model and backbone L2 0/6 both before and after weight matching.

### Possible Interpretations

1. **The fingerprint lives in representation space:** the last task is clearly detectable through CKA and feature drift
2. **Parameter-space distance measures something else:** it consistently selects the first task, i.e. it is dominated by how far the weights travelled in total rather than by recency
3. **No permutation artifact:** weight matching returns the identity permutation, so the L2 result is a property of the solutions, not of neuron numbering
4. **Curvature carries order information:** the Fisher trace is lowest for the most recent task, but the full-order ranking is still near chance
5. **Task similarity effects:** tasks A, B, C may be too similar or too dissimilar

### Limitations

1. **Single training run:** results shown for seed=42 only
2. **Small dataset:** CIFAR-100 has limited examples per class
3. **No task boundaries:** continual learning without explicit task awareness
4. **Six orders only:** with six runs a single flipped prediction changes the accuracy by 16.7 percentage points
5. **Shared initialization:** because all checkpoints share `theta_init`, weight matching cannot show its benefit here
6. **Head architecture fixed:** task heads are always 5-class regardless of cumulative knowledge

### Next Steps for Future Research

1. Vary training seeds and aggregate statistics
2. Repeat weight matching with independently initialized reference models
3. Investigate why parameter distance favors the first task
4. Try alternative model architectures
5. Analyze intermediate layers (not just penultimate)
6. Evaluate with different task combinations and more than three tasks
7. Extend the figure script to the Fisher and weight-matching results
