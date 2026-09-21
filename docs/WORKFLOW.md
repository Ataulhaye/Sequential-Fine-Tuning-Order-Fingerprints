# Sequential Fine-Tuning Order Fingerprints: Complete Workflow

## Overview

This document describes the complete experimental pipeline for studying whether sequential fine-tuning order leaves a measurable fingerprint in neural network weights and representations.

---

## Workflow Steps

### STEP 0: Create the Canonical Base Model

**Script:** `scripts/create_base_model.py`

Run this once before starting any training or analysis:

```bash
python scripts/create_base_model.py
```

This creates and verifies:

```
checkpoints/verified_initialization/base_model.pt
```

The checkpoint contains the untrained model state `theta_0`. Single-task and
sequential training both load this exact state. The command is idempotent: an
existing base checkpoint is preserved and only verified, so rerunning it does
not silently change the initialization used by the experiment.

### STEP 1: Configure Experiment

**File:** `configs/experiment.yaml`

This YAML file centrally defines:

- **Seed:** Global reproducibility seed (42)
- **Tasks:** Three 5-class CIFAR-100 superclass tasks (A, B, C)
- **Model:** ResNet18 architecture, 5 classes per task
- **Training:** Epochs per task, batch size, optimizer settings
- **Sequential orders:** All six permutations to evaluate
- **Representation probe:** Fixed probe configuration
- **Paths:** All checkpoint and result directories

Example configuration excerpt:
```yaml
tasks:
  A: [beaver, dolphin, otter, seal, whale]
  B: [clock, keyboard, lamp, telephone, television]
  C: [bicycle, bus, motorcycle, pickup_truck, train]

experiment:
  sequential_orders:
    - [A, B, C]
    - [A, C, B]
    - [B, A, C]
    - [B, C, A]
    - [C, A, B]
    - [C, B, A]
```

**Note:** Do not hard-code task orders or class lists in scripts. Always read from this configuration.

---

### STEP 2: Train Single-Task Reference Models

**Script:** `scripts/train_single.py`

**Inputs:**
- `configs/experiment.yaml` (configuration)
- CIFAR-100 dataset (downloaded automatically)

**Produces:**
```
checkpoints/single/task_A.pt
checkpoints/single/task_B.pt
checkpoints/single/task_C.pt
```

**Purpose:**

These three checkpoints are the reference models trained independently on each task. They represent the "ground truth" learned representation for each task when trained in isolation.

Each checkpoint contains:
- Trained ResNet18 backbone
- Task-specific classifier head
- Optimizer state (if resuming)
- Metadata (task name, epoch, metrics)

---

### STEP 3: Train All Sequential Orders

**Script:** `scripts/train_sequential.py`

**Inputs:**
- `configs/experiment.yaml` (configuration)
- CIFAR-100 dataset (downloaded automatically)

**Produces:**
```
checkpoints/sequential/A_B_C/A.pt        # After task A
checkpoints/sequential/A_B_C/A_B.pt      # After task B
checkpoints/sequential/A_B_C/A_B_C.pt    # After task C (final)

checkpoints/sequential/A_C_B/A.pt        # After task A
checkpoints/sequential/A_C_B/A_C.pt      # After task C
checkpoints/sequential/A_C_B/A_C_B.pt    # After task B (final)

... (similar structure for all six orders)
```

**Purpose:**

This trains each of the six sequential orders, saving intermediate checkpoints after each task. The naming convention is crucial:

- `A_B_C.pt` means: trained on A, then B, then C (last task = C)
- `A_C_B.pt` means: trained on A, then C, then B (last task = B)
- etc.

Each checkpoint contains:
- Backbone shared across all tasks
- Multiple task-specific heads (one per task learned so far)
- Metadata recording the order and current task

---

### STEP 4: Evaluate Sequential Checkpoints

**Script:** `scripts/evaluate_sequential.py`

**Inputs:**
- `configs/experiment.yaml`
- All sequential checkpoints (from STEP 3)

Note: Single-task checkpoints (from STEP 2) are not used by this script;
they are used later by representation analysis and weight-distance analysis.

**Produces:**
```
results/sequential/A_B_C/metrics.json
results/sequential/A_C_B/metrics.json
results/sequential/B_A_C/metrics.json
results/sequential/B_C_A/metrics.json
results/sequential/C_A_B/metrics.json
results/sequential/C_B_A/metrics.json
```

**Example output structure:**
```json
{
  "order": ["A", "B", "C"],
  "stages": [
    {
      "stage": 1,
      "learned_task": "A",
      "task_history": ["A"],
      "evaluations": [
        {"task": "A", "loss": 1.27, "accuracy": 0.45}
      ]
    },
    {
      "stage": 2,
      "learned_task": "B",
      "task_history": ["A", "B"],
      "evaluations": [
        {"task": "A", "loss": 1.37, "accuracy": 0.42},
        {"task": "B", "loss": 1.56, "accuracy": 0.31}
      ]
    },
    ...
  ],
  "forgetting": {
    "A": 0.004,
    "B": 0.040,
    "C": 0.000
  }
}
```

**Purpose:**

This evaluates the accuracy and loss of each checkpoint on all tasks learned so far, and then computes forgetting metrics.

**Forgetting Definition:**

For each task T:
```
forgetting(T) = 
    accuracy immediately after learning T
    -
    final accuracy on T
```

Positive values indicate performance loss. Negative values indicate performance improvement.

---

### STEP 5: Create Fixed Representation Probe

**Script:** `scripts/create_probe.py`

**Inputs:**
- `configs/experiment.yaml`
- CIFAR-100 test set

**Produces:**
```
results/representation/probe_set.json
```

**Example output structure:**
```json
{
  "seed": 42,
  "samples_per_class": 20,
  "num_classes": 15,
  "num_samples": 300,
  "classes": ["beaver", "dolphin", ..., "train"],
  "indices": {
    "beaver": [15, 42, 87, ...],
    "dolphin": [10, 23, 56, ...],
    ...
  },
  "all_indices": [10, 15, 23, 42, 56, 87, ...]
}
```

**Purpose:**

This creates a deterministic, fixed set of 300 test images (20 per class from all 15 CIFAR-100 classes in tasks A, B, C) using a fixed seed.

**Critical:** This probe must be created ONCE and then reused for all subsequent representation analyses. This ensures that:

1. Single-A, Single-B, and Single-C are evaluated on the exact same images
2. All sequential final checkpoints are compared using the same probe images
3. Results are comparable across all analyses

The probe seed (42) is independent from the global training seed and controls only which specific test images are selected.

---

### STEP 6: Representation Analysis

**Script:** `scripts/analyze_representation.py`

**Inputs:**
- `configs/experiment.yaml`
- Single-task checkpoints
- Sequential final checkpoints
- Frozen probe from STEP 5

**Produces:**
```
results/representation/representation.json
```

**Example output structure:**
```json
{
  "probe": {
    "num_samples": 300,
    "seed": 42
  },
  "orders": [
    {
      "order": ["A", "B", "C"],
      "actual_last_task": "C",
      "checkpoint": "checkpoints/sequential/A_B_C/A_B_C.pt",
      "comparisons": {
        "A": {"cka": 0.731, "feature_drift": 0.464},
        "B": {"cka": 0.378, "feature_drift": 0.550},
        "C": {"cka": 0.338, "feature_drift": 0.539}
      },
      "prediction": {
        "cka": "A",
        "feature_drift": "A"
      },
      "correct": {
        "cka": false,
        "feature_drift": false
      }
    },
    ...
  ]
}
```

**Purpose:**

This calculates representation-level comparisons:

1. **Linear CKA (Centered Kernel Alignment):**
   - Compares the Gram matrices of feature representations
   - High CKA → representations are structurally similar
   - Prediction rule: maximum CKA similarity indicates the last task

2. **Feature Drift:**
   - Measures sample-wise L2 distance between normalized feature vectors
   - Low drift → sequential model's representation is close to reference
   - Prediction rule: minimum drift indicates the last task

Both metrics are computed between the final sequential checkpoint and each single-task reference on the same frozen probe.

---

### STEP 7: Weight-Distance Analysis

**Script:** `scripts/analyze_weight_distance.py`

**Inputs:**
- `configs/experiment.yaml`
- Single-task checkpoints
- Sequential final checkpoints

**Produces:**
```
results/weight_distance/weight_distance.json
```

**Example output structure:**
```json
{
  "analysis": "weight_distance",
  "metrics": ["full_model_l2", "backbone_l2"],
  "orders": [
    {
      "order": ["A", "B", "C"],
      "actual_last_task": "C",
      "checkpoint": "checkpoints/sequential/A_B_C/A_B_C.pt",
      "full_model": {
        "distances": {"A": 2.388, "B": 3.327, "C": 3.352},
        "predicted_last_task": "A",
        "correct": false
      },
      "backbone": {
        "distances": {"A": 2.324, "B": 3.236, "C": 3.257},
        "predicted_last_task": "A",
        "correct": false
      }
    },
    ...
  ],
  "summary": {
    "total_predictions": 6,
    "full_model": {"correct_predictions": 2, "prediction_accuracy": 0.333},
    "backbone": {"correct_predictions": 2, "prediction_accuracy": 0.333}
  }
}
```

**Purpose:**

This calculates parameter-space comparisons:

1. **Full-Model L2 Distance:**
   - Euclidean distance between ALL model parameters (backbone + all heads)
   - Captures the entire parameter state including task-specific heads
   - Prediction rule: minimum distance indicates the last task

2. **Backbone-Only L2 Distance:**
   - Euclidean distance between ONLY ResNet18 backbone parameters
   - Ignores task-specific classification heads
   - Focuses on the shared learned representation
   - Prediction rule: minimum distance indicates the last task

Both metrics compare the final sequential checkpoint directly against each single-task reference.

---

### STEP 8: Loss-Barrier Curve Analysis (Optional)

**Script:** `scripts/analyze_loss_barrier.py`

**Inputs:**
- `configs/experiment.yaml` (configuration)
- `checkpoints/sequential/*/A_B_C.pt` (final checkpoints)
- `checkpoints/single/task_{A,B,C}.pt` (reference models)
- CIFAR-100 dataset (test set)

**Produces:**
```
results/loss_barrier/loss_barrier.json
```

**Purpose:**

Measures the smoothness of the optimization landscape between sequential and single-task models by interpolating in weight space and evaluating loss at the configured number of points.

The number of interpolation points is controlled by `analysis.loss_barrier.num_points` in `configs/experiment.yaml`. Use at least 2 points; larger values produce smoother curves but take longer.

**Analysis:**
- Linear weight-space interpolation: θ(α) = (1-α)θ_ref + α·θ_seq
- Evaluates loss at evenly spaced α values from 0 to 1
- Computes barrier height = max(losses) - min(losses)
- Computes barrier area using trapezoidal integration
- Performed on each task's test set

**Interpretation:**
- **Low barrier** (< 0.1): Similar loss landscapes, compatible solutions
- **High barrier** (> 0.5): Different local minima, divergent trajectories
- **Barrier on last task:** Expected lower (most recent training)
- **Barrier on earlier tasks:** May be higher due to forgetting

**Example output:**
```json
{
  "A_B_C": {
    "order": ["A", "B", "C"],
    "barriers": {
      "Single-A": {
        "A": {
          "alphas": [0.0, 0.1, ..., 1.0],
          "losses": [0.045, 0.062, ..., 0.041],
          "barrier_height": 0.021,
          "barrier_area": 0.082
        },
        ...
      },
      ...
    }
  },
  ...
}
```

---

### STEP 9: Jacobian Sensitivity Analysis (Optional)

**Script:** `scripts/analyze_jacobian.py`

**Inputs:**
- `configs/experiment.yaml` (configuration)
- `checkpoints/sequential/*/A_B_C.pt` (final checkpoints)
- `checkpoints/single/task_{A,B,C}.pt` (reference models)
- `results/representation/probe_set.json` (task probe images)

**Produces:**
```
results/jacobian/jacobian_analysis.json
```

**Purpose:**

Computes input-output Jacobian sensitivity on task probe images to understand how sensitive model predictions are to input perturbations. Reveals task-specific sensitivity patterns and which input features most influence predictions.

**Analysis:**
- For each probe image, compute: J = ∂(predicted_logit)/∂(input)
- Compute sensitivity metrics: mean, max, min, std of Jacobian norms
- Compute spatial sensitivity map: which image regions influence predictions
- Compute channel sensitivity: which color channels carry information
- Performed on both single-task and sequential models

**Note:** This is computationally intensive because the script computes one backward pass per analyzed image. The maximum number of images per model and task is controlled by `analysis.jacobian.max_samples` in `configs/experiment.yaml`. Larger values improve aggregate stability but increase runtime.

**Interpretation:**
- **High sensitivity (mean > 1.0):** Strong input dependence, high feature reliance
- **Low sensitivity (mean < 0.1):** Robust predictions, feature invariance
- **Spatial patterns:** Center/edge/corner preferences in feature importance
- **Channel analysis:** Which colors (R/G/B) encode task information

**Example output:**
```json
{
  "A_B_C": {
    "order": ["A", "B", "C"],
    "jacobians": {
      "Single-A": {
        "model": "Single-A",
        "task": "A",
        "num_samples": 100,
        "metrics": {
          "mean_sensitivity": 0.342,
          "max_sensitivity": 1.241,
          "min_sensitivity": 0.018,
          "std_sensitivity": 0.198,
          "median_sensitivity": 0.289
        },
        "spatial_sensitivity_mean": 0.045,
        "channel_sensitivity": {
          "R": 0.051,
          "G": 0.042,
          "B": 0.048
        }
      },
      ...
    }
  },
  ...
}
```

---

### STEP 10: Combined Analysis

**Script:** `scripts/analyze_combined.py`

**Inputs:**
- `results/sequential/*/metrics.json`
- `results/weight_distance/weight_distance.json`
- `results/representation/representation.json`
- `results/loss_barrier/loss_barrier.json` (if available)
- `results/jacobian/jacobian_analysis.json` (if available)

**Produces:**
```
results/combined/combined_analysis.json
```

**Example output structure:**
```json
{
  "analysis": "combined",
  "orders": [
    {
      "order": ["A", "B", "C"],
      "order_name": "A_B_C",
      "actual_last_task": "C",
      "forgetting": {"A": 0.004, "B": 0.040, "C": 0.000},
      "weight_distance": {
        "full_model": {
          "distances": {"A": 2.388, "B": 3.327, "C": 3.352},
          "predicted_last_task": "A",
          "correct": false
        },
        "backbone": {
          "distances": {"A": 2.324, "B": 3.236, "C": 3.257},
          "predicted_last_task": "A",
          "correct": false
        }
      },
      "representation": {
        "cka": {"A": 0.731, "B": 0.378, "C": 0.338},
        "feature_drift": {"A": 0.464, "B": 0.550, "C": 0.539},
        "predicted_last_task": "A",
        "cka_correct": false,
        "drift_predicted_last_task": "A",
        "drift_correct": false
      },
      "checkpoints": {...}
    },
    ...
  ]
}
```

**Purpose:**

This combines the results from STEP 4, 6, and 7 into a single comprehensive analysis file without recalculating any metrics. When the optional loss-barrier and Jacobian files are present, they are also merged into the same order-level summary. It serves as the primary reference for interpreting the complete experiment.

---

## Data Flow Diagrams

### Training Pipeline

```
configs/experiment.yaml
         |
         +---> train_single.py
         |         |
         |         v
         |     checkpoints/single/{A,B,C}.pt
         |
         +---> train_sequential.py
               |
               v
           checkpoints/sequential/
               {order}/
               {stage}.pt
```

### Representation Probe Pipeline

```
configs/experiment.yaml
         |
         v
    create_probe.py
         |
         v
    probe.py
         |
         v
    results/representation/probe_set.json
         |
         +---> analyze_representation.py
               |
               v
           results/representation/representation.json
```

### Complete Analysis Pipeline

```
configs/experiment.yaml
         |
         +---> evaluate_sequential.py
         |         |
         |         v
         |     results/sequential/{order}/metrics.json
         |
         +---> analyze_weight_distance.py
         |         |
         |         v
         |     results/weight_distance/weight_distance.json
         |
         +---> analyze_representation.py
         |         |
         |         v
         |     results/representation/representation.json
         |
         +---> analyze_loss_barrier.py (optional)
         |         |
         |         v
         |     results/loss_barrier/loss_barrier.json
         |
         +---> analyze_jacobian.py (optional)
         |         |
         |         v
         |     results/jacobian/jacobian_analysis.json
         |
         v
     analyze_combined.py
             |
             v
         results/combined/combined_analysis.json
```

---

## I/O Summary Table

| Script | Reads | Produces | Purpose |
|--------|-------|----------|---------|
| `train_single.py` | config, CIFAR-100 | single checkpoints | Train reference models |
| `train_sequential.py` | config, CIFAR-100 | sequential checkpoints | Train all orders |
| `evaluate_sequential.py` | config, checkpoints | sequential metrics + forgetting | Evaluate accuracy and forgetting |
| `create_probe.py` | config, CIFAR-100 test | probe_set.json | Create deterministic probe |
| `analyze_representation.py` | config, probe, checkpoints | representation.json | Calculate CKA and feature drift |
| `analyze_weight_distance.py` | config, checkpoints | weight_distance.json | Calculate L2 distances |
| `analyze_loss_barrier.py` | config, checkpoints, CIFAR-100 | loss_barrier.json | Calculate loss-barrier curves (optional) |
| `analyze_jacobian.py` | config, probe, checkpoints | jacobian_analysis.json | Calculate Jacobian sensitivity (optional) |
| `analyze_combined.py` | existing result JSON files | combined_analysis.json | Combine core and optional advanced results |
| `plot_combined_results.py` | combined_analysis.json | PNG/PDF figures under `figures/combined/` | Plot research figures and diagnostics |

---

## Test Files

The lightweight checks live in `tests/` rather than the main workflow steps. Use them when changing configuration, data loading, representation extraction, reproducibility logic, or the training loop:

| Test file | Purpose |
|-----------|---------|
| `tests/test_config.py` | Checks that the experiment configuration loads and contains the expected sections. |
| `tests/test_dataset.py` | Checks task dataset construction and task-local labels. |
| `tests/test_representation.py` | Checks representation extraction and pairwise CKA/feature-drift calculations on the fixed probe. |
| `tests/test_reproducibility.py` | Checks deterministic behavior controlled by the project seeds. |
| `tests/smoke_test_training.py` | Runs a small training smoke test to catch basic training-loop failures. |

---

## Quick Start (from scratch)

```bash
# 1. Configure
# Edit configs/experiment.yaml as needed

# 2. Train single-task references
python scripts/train_single.py

# 3. Train all sequential orders
python scripts/train_sequential.py

# 4. Evaluate sequential
python scripts/evaluate_sequential.py

# 5. Create probe (once)
python scripts/create_probe.py

# 6. Analyze representation
python scripts/analyze_representation.py

# 7. Analyze weight distance
python scripts/analyze_weight_distance.py

# 8. Optional: Advanced diagnostics
python scripts/analyze_loss_barrier.py       # Loss-barrier curves
python scripts/analyze_jacobian.py           # Jacobian sensitivity

# 9. Combine all results
python scripts/analyze_combined.py

# 10. Generate report figures
python scripts/plot_combined_results.py
```

The plotting script reads `results/combined/combined_analysis.json` and writes the research figure suite under `figures/combined/`, including prediction accuracy, prediction matrices, raw score plots, forgetting, loss-barrier diagnostics, Jacobian diagnostics, and a compact research summary. See `docs/FIGURES.md` for a visual guide to the generated plots.

Optional checks after setup or code changes:

```bash
python tests/test_config.py
python tests/test_dataset.py
python tests/test_representation.py
python tests/test_reproducibility.py
python tests/smoke_test_training.py
```

---

## Important Notes

1. **Fixed Probe:** The representation probe must be created once and then reused. Do not recreate it unless explicitly intended.

2. **Sequential Orders:** Always read sequential orders from `config["experiment"]["sequential_orders"]`. Never hard-code them.

3. **Reproducibility:** The global seed and representation probe seed are independent:
   - `seed: 42` controls training reproducibility
   - `representation.probe.seed: 42` controls which test images are selected

4. **Task Naming:** Sequential checkpoints use underscore-separated format: `A_B_C.pt`, `A_C_B.pt`, etc.

5. **Last Task:** Always use `order[-1]` to determine the actual last task for prediction evaluation.

6. **Results Location:** Results are saved under `results/` with subdirectories for each analysis type.
