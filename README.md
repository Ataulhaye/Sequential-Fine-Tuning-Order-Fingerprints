# Sequential Fine-Tuning Order Fingerprints

This project investigates whether a neural network trained on multiple tasks in sequence retains a measurable geometric fingerprint revealing the order in which tasks were learned.

**Quick Links:**
- **[WORKFLOW.md](docs/WORKFLOW.md)** — Complete step-by-step execution guide
- **[METHODS.md](docs/METHODS.md)** — Detailed scientific methodology
- **[configs/experiment.yaml](configs/experiment.yaml)** — Central configuration file

## 1. Research Question

> **Given the final checkpoint of a sequentially fine-tuned model on three tasks, can we determine which task was learned most recently?**

For example, if we sequentially trained a model as:
```
Task A → Task B → Task C
```

Can we predict that C is the last task by comparing this final model against three independently trained reference models (Single-A, Single-B, Single-C)?

## 2. Core Concept

This project uses **six sequential task orders** (all permutations of three tasks A, B, C):

| Order 1 | Order 2 | Order 3 | Order 4 | Order 5 | Order 6 |
|---------|---------|---------|---------|---------|---------|
| A → B → C | A → C → B | B → A → C | B → C → A | C → A → B | C → B → A |

For each order, we:
1. Train sequentially on all three tasks
2. Save checkpoints after each task
3. Compare the final checkpoint against three single-task references
4. Test if we can predict the last task

## 3. Tasks: CIFAR-100 Superclasses

Three 5-class CIFAR-100 superclass-based tasks:

**Task A (Aquatic Animals)**
```
beaver, dolphin, otter, seal, whale
```

**Task B (Electronic Devices)**
```
clock, keyboard, lamp, telephone, television
```

**Task C (Vehicles)**
```
bicycle, bus, motorcycle, pickup_truck, train
```

Classes are defined in `configs/experiment.yaml`; the main training and analysis
pipelines read them from the configuration.

## 4. Model Architecture

**Backbone:** ResNet18 (shared across all tasks)

**Task Heads:** One 5-class classification head per task

```
Input Images
    ↓
ResNet18 Backbone (Shared)
    ↓
    ├── Head A (5-class) → Predictions on Task A
    ├── Head B (5-class) → Predictions on Task B
    └── Head C (5-class) → Predictions on Task C
```

The shared backbone is the primary object of study, representing the learned feature space.

## 5. Three Comparison Methods

The project compares sequential final models against single-task references using three independent approaches:

### 5.1 Weight Distance

**Full-Model L2:** Euclidean distance between all parameters
- Captures complete model divergence
- Includes backbone + all heads

**Backbone-Only L2:** Euclidean distance between only backbone parameters
- Isolates shared representation changes
- Ignores task-specific heads

**Prediction Rule:** Minimum distance indicates last task

### 5.2 Representation Similarity (CKA)

**Linear CKA:** Centered Kernel Alignment of feature activations
- Compares learned representations on fixed probe
- Invariant to orthogonal transformations
- High CKA → structurally similar representations

**Prediction Rule:** Maximum CKA similarity indicates last task

### 5.3 Feature Drift

**Definition:** Sample-wise L2 distance between normalized features
- Measures how individual feature vectors changed
- Computed on the same fixed probe as CKA
- Low drift → representation unchanged

**Prediction Rule:** Minimum drift indicates last task

### Method Comparison

| Method | Space | Metric | Prediction |
|--------|-------|--------|-----------|
| Full-model L2 | Parameters | Distance | Min distance |
| Backbone L2 | Parameters | Distance | Min distance |
| CKA | Representations | Similarity | Max similarity |
| Feature Drift | Representations | Distance | Min distance |

## 6. Fixed Representation Probe

To ensure fair representation comparisons across all models, a **fixed, deterministic probe** is created once and reused:

- **Composition:** 20 images per class × 15 classes = 300 images
- **Classes:** All classes from tasks A, B, and C
- **Seed:** 42 (fixed and independent from training seed)
- **Source:** CIFAR-100 test set
- **Reused by:** CKA analysis, feature drift analysis, sanity checks

The probe is created with `python scripts/create_probe.py` and stored in `results/representation/probe_set.json`.

**Critical:** Never recreate the probe unless explicitly intended. Reuse ensures that all models are evaluated on the identical set of images, making results comparable.

## 7. Forgetting Metric

**Forgetting Definition:**
```
forgetting(T) = accuracy(T) immediately after learning T
                -
                accuracy(T) at end of training
```

Positive values indicate performance loss on earlier tasks. Measured during sequential evaluation.

## 8. Current Results (Smoke Test, 1 Epoch)

### Prediction Accuracy (All Methods)

| Method | Correct | Total | Accuracy |
|--------|---------|-------|----------|
| Full-Model L2 | 2 | 6 | 33.33% |
| Backbone-Only L2 | 2 | 6 | 33.33% |
| CKA | 2 | 6 | 33.33% |
| Feature Drift | 2 | 6 | 33.33% |

**Note:** 33.33% is random-chance accuracy (1 out of 3 tasks).

### Key Observation

All methods currently predict task **A** as the last task for most orders, regardless of actual ground truth. This suggests:

1. The task order may not leave a strong geometric fingerprint in the current setup
2. Single-epoch training may be insufficient
3. Alternative metrics or model architectures might be needed

## 8.5 Advanced Diagnostics

### 8.5.1 Loss-Barrier Curves

Measures the "smoothness" of the optimization landscape between sequential and single-task models by interpolating in weight space.

- **Low barrier:** Similar loss landscapes → comparable learned solutions
- **High barrier:** Different local minima → divergent learning trajectories
- **Computation:** Linear interpolation between weight vectors; the number of evaluation points is configured by `analysis.loss_barrier.num_points`
- **Output:** Barrier height, area under curve, per-task analysis

### 8.5.2 Jacobian Sensitivity Analysis

Computes input-output Jacobian sensitivity to understand how sensitive model predictions are to input perturbations.

The number of analyzed images is configured by `analysis.jacobian.max_samples` in `configs/experiment.yaml`.

- **Mean Sensitivity:** Average Jacobian norm (high = strong input dependence)
- **Spatial Sensitivity:** Mean spatial sensitivity across the image (`spatial_sensitivity_mean`); the current JSON output does not store the full map
- **Channel Sensitivity:** Which color channels (R/G/B) carry most information
- **Task Comparison:** Reveals differences in feature importance across tasks

See [METHODS.md](docs/METHODS.md) Sections 12-13 for detailed methodology.

## 9. Project Structure

```
Sequential Fine-Tuning Order Fingerprints/
│
├── README.md                          # This file
├── docs/
│   ├── WORKFLOW.md                    # Step-by-step execution guide
│   └── METHODS.md                     # Detailed scientific methodology
│
├── configs/
│   └── experiment.yaml                # Central configuration
│
├── src/sequential_finetuning/
│   ├── config.py                      # Config loading
│   ├── dataset.py                     # CIFAR-100 task dataloaders
│   ├── model.py                       # ResNet18MultiTask architecture
│   ├── train.py                       # Training loop
│   ├── checkpoint.py                  # Save/load checkpoints
│   ├── evaluation.py                  # Task evaluation
│   ├── forgetting.py                  # Forgetting calculation
│   ├── seed.py                        # Reproducibility
│   ├── training_utils.py              # Utility functions
│   ├── probe.py                       # Fixed representation probe
│   ├── representation.py              # CKA, feature drift, extraction
│   ├── results.py                     # Results formatting
│   ├── combined_analysis.py           # Result combination
│   └── analysis/
│       ├── weight_distance.py         # L2 distance calculations
│       ├── loss_barrier.py            # Loss-barrier curve analysis
│       └── jacobian.py                # Jacobian sensitivity analysis
│
├── scripts/
│   ├── train_single.py                # Train single-task models
│   ├── train_sequential.py            # Train all six sequential orders
│   ├── evaluate_sequential.py         # Evaluate and calculate forgetting
│   ├── create_probe.py                # Create fixed representation probe
│   ├── test_representation.py         # Sanity check representations
│   ├── analyze_representation.py      # Calculate CKA and feature drift
│   ├── analyze_weight_distance.py     # Calculate L2 distances
│   ├── analyze_loss_barrier.py        # Loss-barrier curve analysis
│   ├── analyze_jacobian.py            # Jacobian sensitivity analysis
│   ├── analyze_combined.py            # Combine all results
│   └── smoke_test_training.py         # Quick local test
│
├── checkpoints/
│   ├── single/                        # Single-task reference models
│   │   ├── task_A.pt
│   │   ├── task_B.pt
│   │   └── task_C.pt
│   └── sequential/                    # Sequential order checkpoints
│       ├── A_B_C/
│       ├── A_C_B/
│       ├── B_A_C/
│       ├── B_C_A/
│       ├── C_A_B/
│       └── C_B_A/
│
├── results/
│   ├── sequential/                    # Sequential evaluation results
│   │   ├── A_B_C/metrics.json
│   │   ├── A_C_B/metrics.json
│   │   └── ... (one per order)
│   ├── weight_distance/               # Weight distance analysis
│   │   └── weight_distance.json
│   ├── representation/                # Representation analysis
│   │   ├── probe_set.json
│   │   └── representation.json
│   ├── loss_barrier/                  # Optional loss-barrier analysis
│   │   └── loss_barrier.json
│   ├── jacobian/                      # Optional Jacobian analysis
│   │   └── jacobian_analysis.json
│   └── combined/                      # Combined results
│       └── combined_analysis.json
│
└── data/
    └── cifar-100-python/              # CIFAR-100 dataset (auto-downloaded)
```

## 10. Execution Workflow

For complete details, see **[docs/WORKFLOW.md](docs/WORKFLOW.md)**.

**Quick summary:**

```bash
# 1. Configure (edit configs/experiment.yaml if needed)

# 2. Train single-task references
python scripts/train_single.py

# 3. Train all sequential orders
python scripts/train_sequential.py

# 4. Evaluate sequential training
python scripts/evaluate_sequential.py

# 5. Create fixed probe (once)
python scripts/create_probe.py

# 6. Optional: sanity check
python scripts/test_representation.py

# 7. Analyze representations (CKA, drift)
python scripts/analyze_representation.py

# 8. Analyze weight distances
python scripts/analyze_weight_distance.py

# 9. Advanced diagnostics (optional)
python scripts/analyze_loss_barrier.py       # Loss-barrier curves
python scripts/analyze_jacobian.py           # Jacobian sensitivity

# 10. Combine all results
python scripts/analyze_combined.py

# 11. Generate report figures
python scripts/plot_combined_results.py
```

## 11. Configuration: experiment.yaml

The single source of truth for all experiments.

**Key sections:**

- `seed: 42` — Global training reproducibility seed
- `tasks` — Task class definitions (A, B, C)
- `model` — ResNet18 configuration
- `training` — Epochs, batch size, learning rate, etc.
- `experiment.sequential_orders` — Six task orders
- `representation.probe` — Fixed probe configuration
- `analysis.loss_barrier.num_points` — Loss-barrier interpolation resolution
- `analysis.jacobian.max_samples` — Maximum Jacobian images per model/task
- `paths` — Checkpoint and result directories

**Important:** Main training and analysis pipelines must read task orders and
classes from this configuration rather than duplicating experiment definitions.

## 12. Main Results Files

| File | Description |
|------|-------------|
| `results/sequential/{order}/metrics.json` | Accuracy and forgetting for each order |
| `results/weight_distance/weight_distance.json` | Full-model and backbone L2 distances |
| `results/representation/representation.json` | CKA and feature drift comparisons |
| `results/representation/probe_set.json` | Fixed probe definition |
| `results/loss_barrier/loss_barrier.json` | Optional loss-barrier curves and summary metrics |
| `results/jacobian/jacobian_analysis.json` | Optional Jacobian sensitivity metrics and channel summaries |
| `results/combined/combined_analysis.json` | Combined results from all core methods and optional advanced diagnostics |
| `figures/combined/` | Core and, when available, advanced diagnostic PNG figures |

## 13. Reproducibility

Two independent seeds control reproducibility:

1. **Global Seed (`seed: 42`):** Controls training initialization, weight initialization, data shuffling
2. **Representation Probe Seed (`representation.probe.seed: 42`):** Controls which CIFAR-100 test images are selected

This separation allows:
- Reusing a fixed probe across different training runs
- Changing training seed without recomputing probe
- Systematic variation while maintaining comparability

## 14. Methodology Details

For in-depth explanation of:
- Weight-space comparison (full-model vs. backbone L2)
- Representation-space comparison (CKA vs. feature drift)
- Forgetting calculation
- Checkpoint conventions
- Last-task prediction rules

See **[docs/METHODS.md](docs/METHODS.md)**.

## 15. Current Status

**Development Stage:** Smoke test with 1 epoch per task (local verification)

**Current implementation status:**
- Core analyses are fully implemented and combined
- Optional advanced diagnostics (loss-barrier curves, Jacobian sensitivity) are supported and merged when available
- Combined JSON includes all available methods without recomputing metrics

**Next Steps:**
1. Full training on university server (10-20 epochs per task)
2. Multiple random seeds for statistical robustness
3. Analysis of intermediate layers
4. Investigation of task similarity effects
5. Alternative model architectures

## 16. Key Findings from Current Smoke Test

### Observation 1: Weak or Absent Fingerprint

All four methods predict task A as the last task for most orders, achieving 33.33% accuracy (random chance).

### Observation 2: Consistent Bias

The bias toward predicting A suggests either:
- Task A has some structural/representational advantage in the current setup
- Fingerprint effects require deeper networks or more training
- The signal is too weak for 1-epoch training

### Observation 3: Forgetting Patterns

Forgetting values are generally small (< 0.05), suggesting minimal performance degradation despite sequential training and no explicit regularization.

## 17. Scientific Integrity Notes

- All results shown are from **observed experiment runs**, not theoretical predictions
- Accuracy values (33.33%) reflect **current smoke test** with 1 epoch
- Claims of "unsuccessful" methods are **not made**; rather, methods show **no clear advantage** in current setup
- The project aims to investigate whether fingerprints exist, not to demonstrate that they do
- Full-experiment results on university server will be necessary for statistical conclusions

## 18. Contributing

To extend this project:

1. **New comparison method?** Implement in `src/sequential_finetuning/` and create corresponding analysis script
2. **New metric?** Add to configuration and corresponding analysis module
3. **New task sets?** Update `configs/experiment.yaml`
4. **New model architecture?** Modify `src/sequential_finetuning/model.py` and update checkpoint loading

Remember:
- Never hard-code task orders or class lists
- Always read from `configs/experiment.yaml`
- Save intermediate results to JSON files
- Document inputs/outputs in script docstrings
- Use centralized probe for representation analyses

## 19. References

**Files Referenced in This README:**
- Configuration: [configs/experiment.yaml](configs/experiment.yaml)
- Workflow: [docs/WORKFLOW.md](docs/WORKFLOW.md)
- Methods: [docs/METHODS.md](docs/METHODS.md)
- Main Module: [src/sequential_finetuning/](src/sequential_finetuning/)
