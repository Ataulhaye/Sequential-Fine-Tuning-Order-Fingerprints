# Project 3: Sequential Fine-Tuning Order Fingerprints

## 1. Project Overview

This project investigates whether a neural network that has been **sequentially fine-tuned on multiple tasks** retains a measurable geometric fingerprint of the **most recently learned task**.

The central research question is:

> **Given only the final checkpoint of a sequentially fine-tuned model, can we determine which task was learned last?**

We will train ResNet18 models on three CIFAR-100 superclass-based classification tasks using different task orders. The final sequentially fine-tuned models will then be compared with independently trained single-task checkpoints using several measures of similarity.

The main comparison methods are:

1. **Weight distance**
2. **Representation similarity using CKA**
3. **Feature drift**
4. **Task accuracy and forgetting**

Additional analyses will investigate whether geometric similarity is related to forgetting:

5. **Loss-barrier curves**
6. **Optional input-output Jacobian sensitivity**

The project will initially be developed and debugged locally using a single-epoch smoke test. Full experiments will subsequently be executed on the university server using multiple epochs and multiple random seeds.

---

# 2. Research Question

The primary question is:

> **Does the final checkpoint of a sequentially fine-tuned model reveal which task was learned most recently?**

For example, consider three tasks:

```text
Task A
Task B
Task C
```

A sequential model may be trained as:

```text
A → B → C
```

The final checkpoint is then compared with independently trained models:

```text
Single-A
Single-B
Single-C
```

If the final `A → B → C` model is consistently most similar to `Single-C`, this would suggest that learning C last leaves a measurable fingerprint in the final network.

We will test whether this effect occurs consistently across different task orders and random seeds.

---

# 3. Main Hypotheses

## H1 — Last-task fingerprint

The final sequentially fine-tuned model will be more similar to the single-task model corresponding to the **most recently learned task** than to the earlier tasks.

For example:

```text
A → B → C
```

should ideally produce:

```text
Similarity(final_ABC, Single-C)
    >
Similarity(final_ABC, Single-B)
    >
Similarity(final_ABC, Single-A)
```

The exact ordering beyond the last task is not assumed beforehand.

---

## H2 — Representation similarity will reveal task order

Representation-level similarity, particularly CKA, will identify the most recently learned task more reliably than raw parameter-space distance.

This hypothesis tests whether the geometric structure of the learned representation contains a stronger task-order signal than the raw weights.

---

## H3 — Earlier tasks experience greater forgetting

Tasks learned earlier in the sequence are expected to experience greater accuracy degradation and/or feature drift than the task learned most recently.

For:

```text
A → B → C
```

we expect, in general:

```text
Forgetting(A) ≥ Forgetting(B) ≥ Forgetting(C)
```

although this is an empirical hypothesis rather than a guaranteed outcome.

---

## H4 — Feature drift is related to forgetting

Larger changes in a task's representation after subsequent fine-tuning may be associated with larger decreases in that task's accuracy.

We will test whether representation drift provides a useful diagnostic of forgetting.

---

# 4. Experimental Concept

The experiment consists of two types of models.

## 4.1 Single-task models

Each task is trained independently from the same initial ResNet18 configuration.

```text
Base
 ├──→ Task A → Single-A
 ├──→ Task B → Single-B
 └──→ Task C → Single-C
```

These models serve as reference checkpoints.

---

## 4.2 Sequential models

The same tasks are learned sequentially in different orders.

With three tasks there are six possible orders:

```text
A → B → C
A → C → B
B → A → C
B → C → A
C → A → B
C → B → A
```

Each sequence produces a final checkpoint.

Intermediate checkpoints will also be saved.

For example:

```text
A → B → C

after A     → ABC_step1
after A,B   → ABC_step2
after A,B,C → ABC_final
```

Saving intermediate checkpoints is essential for measuring feature drift and forgetting.

---

# 5. CIFAR-100 Task Construction

CIFAR-100 contains:

- 100 fine-grained classes
- 20 superclasses
- 5 fine-grained classes per superclass

Each experimental task will be constructed from one CIFAR-100 superclass.

Therefore each task will be a 5-class classification problem.

Conceptually:

```text
Task A
 ├── class A1
 ├── class A2
 ├── class A3
 ├── class A4
 └── class A5

Task B
 ├── class B1
 ├── class B2
 ├── class B3
 ├── class B4
 └── class B5

Task C
 ├── class C1
 ├── class C2
 ├── class C3
 ├── class C4
 └── class C5
```

The exact three superclasses will be selected during implementation.

The task definitions will be stored explicitly in the project configuration so that all experiments use exactly the same class assignments.

---

# 6. Model Architecture

The primary model will be:

**ResNet18**

with an appropriate CIFAR-100 input configuration.

The main object of study is the shared feature extractor/backbone.

Each task will use a task-specific classification head:

```text
                    ┌── Head A
                    │
Input → ResNet18 ───┼── Head B
                    │
                    └── Head C
```

The task-specific heads allow us to focus the analysis on changes to the learned representation rather than confusing representation changes with incompatible output-label spaces.

The exact implementation of the task heads will be documented in the code and experiment configuration.

---

# 7. Single-Task Baseline Training

Before sequential training, three independent single-task models will be trained.

```text
Single-A
Single-B
Single-C
```

Each model will start from the same ResNet18 architecture and independently initialized parameters according to the experiment seed.

The single-task checkpoints provide the reference points for all subsequent comparisons.

For each single-task model we will record:

- training loss
- validation/test loss
- task accuracy
- checkpoint
- model configuration
- random seed
- training duration

---

# 8. Sequential Fine-Tuning

Each sequential experiment will start from the appropriate initial model and learn the three tasks one after another.

Example:

```text
Initial model
     ↓
   Task A
     ↓
Checkpoint A
     ↓
   Task B
     ↓
Checkpoint AB
     ↓
   Task C
     ↓
Checkpoint ABC
```

For each order we will save:

1. Initial checkpoint
2. Checkpoint after task 1
3. Checkpoint after task 2
4. Final checkpoint after task 3

This allows analysis of both:

- final task-order fingerprints
- evolution of representations over time

---

# 9. Task Orders

All six permutations will be evaluated.

| Order ID | Training order |
|---|---|
| ABC | A → B → C |
| ACB | A → C → B |
| BAC | B → A → C |
| BCA | B → C → A |
| CAB | C → A → B |
| CBA | C → B → A |

For each order, the final model will be compared with:

```text
Single-A
Single-B
Single-C
```

The correct last task is known from the training order and will therefore provide the ground truth for the prediction experiment.

---

# 10. Primary Analysis 1 — Weight Distance

The first comparison will operate directly in parameter space.

For two models with parameters:

\[
\theta_1,\theta_2
\]

we compute a normalized parameter distance such as:

\[
D_W(\theta_1,\theta_2)
=
\|\theta_1-\theta_2\|_2
\]

A normalized version may also be evaluated to account for model scale.

For every final sequential model:

```text
Final-ABC
```

we calculate:

```text
distance(Final-ABC, Single-A)
distance(Final-ABC, Single-B)
distance(Final-ABC, Single-C)
```

The closest model becomes the predicted last task.

Example:

```text
          Weight distance

Single-A       120
Single-B        90
Single-C        55   ← predicted last task
```

The analysis will determine whether the minimum distance corresponds to the actual final task.

---

# 11. Primary Analysis 2 — CKA Representation Similarity

Raw weights do not necessarily provide a good measure of functional similarity.

Therefore we will compare internal representations.

Given the same probe dataset, representations will be extracted from selected ResNet18 layers.

For example:

```text
Input
  ↓
Conv / Layer 1
  ↓
Layer 2
  ↓
Layer 3
  ↓
Layer 4
  ↓
Classifier
```

For each selected layer, we will calculate CKA between:

```text
Final sequential model
```

and

```text
Single-task reference model
```

using the same probe images.

The result will be a similarity matrix.

Example:

| Final model | Single-A | Single-B | Single-C |
|---|---:|---:|---:|
| ABC | 0.71 | 0.76 | **0.89** |
| ACB | 0.70 | **0.91** | 0.74 |

The task with the highest CKA similarity will be used as the predicted most-recent task.

CKA will initially be evaluated at selected ResNet18 layers rather than only at the final representation.

---

# 12. Probe Dataset

Representation comparisons require identical inputs.

Therefore a fixed probe dataset will be created.

The same images will be passed through all models being compared.

Conceptually:

```text
                 Same probe images
                       │
          ┌────────────┼────────────┐
          ↓            ↓            ↓
       Single-A     Single-B      Final
          ↓            ↓            ↓
      Features A   Features B   Features Final
          └────────────┼────────────┘
                       ↓
                      CKA
```

Probe indices and configuration will be stored so the experiment is reproducible.

The probe set will not change between model comparisons within a given experiment.

---

# 13. Primary Analysis 3 — Feature Drift

Feature drift measures how much a task's representation changes after subsequent tasks are learned.

For example:

```text
A
↓
A → B
↓
A → B → C
```

Representations for the same task probe images can be compared between these checkpoints.

We can measure:

\[
Drift(A,AB)
\]

and:

\[
Drift(A,ABC)
\]

This allows us to investigate whether early tasks undergo progressively larger representational changes.

Feature drift may be quantified using representation distances and/or CKA-derived dissimilarity:

\[
Drift = 1 - CKA
\]

depending on the final analysis design.

---

# 14. Forgetting Analysis

Forgetting will be measured using task accuracy.

For a task \(T\):

\[
F_T =
Acc_T^{\text{best/prior}}
-
Acc_T^{\text{final}}
\]

For example:

```text
A → B → C

After A:
Accuracy(A) = 91%

After B:
Accuracy(A) = 74%

After C:
Accuracy(A) = 65%
```

Then:

```text
A forgetting = 91 - 65 = 26 percentage points
```

Accuracy will be evaluated after each sequential training stage on all relevant tasks.

This produces a forgetting matrix.

Example:

| Checkpoint | Task A | Task B | Task C |
|---|---:|---:|---:|
| After A | 91 | — | — |
| After A→B | 74 | 90 | — |
| After A→B→C | 65 | 76 | 91 |

This will allow us to compare forgetting with feature drift.

---

# 15. Linking Fingerprints to Forgetting

One of the main scientific analyses will ask:

> **Is the geometric trace of the most recent task related to the forgetting of earlier tasks?**

We will compare:

- weight distance
- CKA similarity
- feature drift
- task accuracy
- accuracy drop

Possible relationships include:

```text
More feature drift
        ↓
More forgetting
```

and:

```text
Higher similarity to last task
        ↓
Stronger last-task fingerprint
```

These relationships will be evaluated empirically rather than assumed.

---

# 16. Secondary Analysis — Loss Barriers

We will optionally investigate the loss landscape between sequential and single-task solutions.

Given two model checkpoints:

```text
Model A
Model B
```

we can examine interpolated parameters:

\[
\theta(\alpha)
=
(1-\alpha)\theta_A+\alpha\theta_B
\]

for:

\[
\alpha \in [0,1]
\]

and evaluate loss along the interpolation path.

Example:

```text
Loss

  ^
  |          /\
  |         /  \
  |________/    \________
  |
  +----------------------→ α
  0                      1
```

This can reveal whether sequential and single-task solutions are connected by low-loss paths or separated by barriers.

Loss-barrier analysis will be treated as a secondary experiment because it is more computationally expensive and requires careful interpretation.

---

# 17. Optional Analysis — Input-Output Jacobian

If time and compute resources permit, we will investigate input-output Jacobian sensitivity on task probe images.

For a model \(f(x)\), we can examine:

\[
J(x)=\frac{\partial f(x)}{\partial x}
\]

This measures how sensitive the output is to changes in the input.

The analysis may compare Jacobian norms or related sensitivity measures across:

- single-task models
- sequential intermediate checkpoints
- sequential final checkpoints

This analysis is optional and will only be implemented after the primary experiments are functioning correctly.

---

# 18. Last-Task Prediction

The core classification experiment is:

> Given a final sequential checkpoint and the three single-task reference checkpoints, predict which task was learned last.

For every metric:

### Weight distance

Choose:

\[
\arg\min_T D_W(Final, Single_T)
\]

### CKA

Choose:

\[
\arg\max_T CKA(Final, Single_T)
\]

### Other representation metrics

Use the corresponding similarity/distance criterion.

The prediction is compared against the known final task.

---

# 19. Evaluation Metric

The primary evaluation metric is:

**Last-task identification accuracy**

For example, if six task orders are tested:

```text
ABC → predicts C ✓
ACB → predicts B ✓
BAC → predicts C ✓
BCA → predicts A ✓
CAB → predicts B ✓
CBA → predicts A ✓
```

then:

```text
6 / 6 = 100%
```

A random three-way classifier has an expected baseline of:

\[
1/3 \approx 33.3\%
\]

Therefore we will compare our methods against this baseline.

With multiple random seeds, prediction accuracy will be reported across runs rather than relying on one experiment.

---

# 20. Multiple Random Seeds

Initial debugging will use a single seed.

Full experiments will use multiple seeds, for example:

```text
seed 0
seed 1
seed 2
```

The final experiment will therefore contain:

```text
6 task orders × N seeds
```

For example, with three seeds:

```text
6 × 3 = 18 sequential experiments
```

This allows us to determine whether the observed task-order fingerprint is robust to random initialization and training variation.

---

# 21. Local Development vs University Server

## Local machine

The local NVIDIA GPU has approximately 8 GB dedicated VRAM.

Local experiments will be used for:

- debugging
- verifying dataset construction
- verifying model construction
- checking checkpoint saving/loading
- testing feature extraction
- testing CKA
- testing distance calculations
- testing evaluation
- one-epoch smoke tests

The local experiments are **not intended to provide the final scientific results**.

A typical smoke test will use approximately:

```text
1 epoch
1 seed
1 task order
```

and potentially reduced dataset/probe sizes where appropriate.

---

## University server

The university server will be used for the full experiment.

The full experiment will use:

- complete CIFAR-100 task datasets
- appropriate training duration
- all six task orders
- multiple random seeds
- all required checkpoints
- representation extraction
- CKA
- forgetting analysis
- feature drift
- loss-barrier analysis where feasible
- optional Jacobian analysis

The exact number of epochs and seeds will be finalized after the local pipeline has been validated.

---

# 22. Reproducibility

Every experiment should record:

- random seed
- task definitions
- task order
- model architecture
- optimizer
- learning rate
- batch size
- number of epochs
- dataset configuration
- data augmentation
- checkpoint path
- software environment
- experiment configuration

Experiments should be reproducible from configuration files rather than manually changing Python source code.

---

# 23. Proposed Project Structure

```text
project3/
│
├── README.md
│
├── requirements.txt
│
├── configs/
│   └── experiment.yaml
│
├── data/
│
├── checkpoints/
│   ├── single/
│   │   ├── task_a/
│   │   ├── task_b/
│   │   └── task_c/
│   │
│   └── sequential/
│       ├── ABC/
│       ├── ACB/
│       ├── BAC/
│       ├── BCA/
│       ├── CAB/
│       └── CBA/
│
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── representations.py
│   ├── cka.py
│   ├── distances.py
│   ├── forgetting.py
│   ├── loss_barrier.py
│   └── jacobian.py
│
├── scripts/
│   ├── train_single.py
│   ├── train_sequential.py
│   ├── evaluate_models.py
│   └── run_analysis.py
│
├── results/
│   ├── checkpoints.csv
│   ├── accuracy.csv
│   ├── weight_distance.csv
│   ├── cka.csv
│   ├── feature_drift.csv
│   ├── forgetting.csv
│   └── predictions.csv
│
├── figures/
│   ├── accuracy/
│   ├── cka/
│   ├── drift/
│   ├── forgetting/
│   └── loss_barriers/
│
└── notebooks/
    └── analysis.ipynb
```

The structure may evolve during implementation.

---

# 24. Expected Outputs

At the end of the main experiment we should have:

### Checkpoints

```text
3 single-task checkpoints
+
6 × N sequential experiment checkpoints
```

including intermediate sequential checkpoints.

### Accuracy results

Accuracy of every checkpoint on every task.

### Weight-distance results

Distances between final sequential models and single-task models.

### CKA results

Layer-wise representation similarities.

### Feature-drift results

Representation changes across sequential training stages.

### Forgetting results

Accuracy degradation for previously learned tasks.

### Prediction results

For every final checkpoint:

```text
Actual last task
Predicted last task
Weight-distance prediction
CKA prediction
Other metric predictions
```

---

# 25. Planned Figures

The final project should include figures such as:

### Figure 1 — Experimental setup

```text
Single-task models
        ↓
Reference checkpoints

Sequential models
        ↓
Different task orders
        ↓
Final checkpoints
        ↓
Similarity analysis
```

### Figure 2 — Accuracy over sequential training

Accuracy of each task as new tasks are learned.

### Figure 3 — Weight-distance matrix

Final sequential models versus single-task checkpoints.

### Figure 4 — CKA similarity matrix

Representation similarity at multiple ResNet18 layers.

### Figure 5 — Feature drift

Representation change for each task as subsequent tasks are learned.

### Figure 6 — Forgetting versus feature drift

Relationship between representational change and accuracy loss.

### Figure 7 — Last-task prediction accuracy

Comparison of:

- weight distance
- CKA
- feature drift / related metric

against the random 33.3% baseline.

### Figure 8 — Loss barriers

Optional loss interpolation curves.

---

# 26. Experimental Milestones

## Milestone 1 — Project setup

- [ ] Create repository
- [ ] Create Python environment
- [ ] Install dependencies
- [ ] Create project structure
- [ ] Define configuration system

## Milestone 2 — Dataset

- [ ] Download CIFAR-100
- [ ] Define three superclasses
- [ ] Implement task datasets
- [ ] Verify class distributions
- [ ] Create fixed probe set

## Milestone 3 — Model

- [ ] Implement ResNet18
- [ ] Implement task-specific heads
- [ ] Verify forward pass
- [ ] Verify checkpoint save/load

## Milestone 4 — Local smoke test

- [ ] Train one task for one epoch
- [ ] Verify loss decreases / training functions
- [ ] Verify evaluation
- [ ] Verify checkpoint loading

## Milestone 5 — Single-task baselines

- [ ] Train Task A
- [ ] Train Task B
- [ ] Train Task C
- [ ] Save checkpoints
- [ ] Record metrics

## Milestone 6 — Sequential training

- [ ] Implement sequential fine-tuning
- [ ] Save intermediate checkpoints
- [ ] Test one task order locally
- [ ] Test all six orders locally

## Milestone 7 — Full university experiments

- [ ] Transfer project to university server
- [ ] Run full training
- [ ] Run multiple seeds
- [ ] Verify all checkpoints

## Milestone 8 — Weight analysis

- [ ] Implement weight distance
- [ ] Generate distance matrices
- [ ] Predict last task
- [ ] Evaluate prediction accuracy

## Milestone 9 — Representation analysis

- [ ] Implement feature extraction
- [ ] Implement CKA
- [ ] Analyze selected layers
- [ ] Predict last task using CKA

## Milestone 10 — Forgetting

- [ ] Calculate task accuracy after each stage
- [ ] Calculate forgetting
- [ ] Calculate feature drift
- [ ] Compare drift with forgetting

## Milestone 11 — Secondary analyses

- [ ] Implement loss barriers
- [ ] Evaluate interpolation curves
- [ ] Implement Jacobian analysis if feasible

## Milestone 12 — Final analysis

- [ ] Aggregate multiple seeds
- [ ] Statistical analysis
- [ ] Generate final figures
- [ ] Interpret results
- [ ] Compare findings with literature
- [ ] Write final report

---

# 27. Literature

The project is primarily motivated by work on continual learning, representation accumulation, feature forgetting, and loss connectivity.

### Hess et al.

**Knowledge Accumulation in Continually Learned Representations and the Issue of Feature Forgetting**

https://arxiv.org/abs/2304.00933

This work motivates the investigation of how representations evolve during continual learning and how feature forgetting can occur.

### Mirzadeh et al.

**Linear Mode Connectivity in Multitask and Continual Learning**

https://arxiv.org/abs/2010.04495

This work motivates the investigation of connectivity and loss landscapes between solutions obtained through multitask and continual learning.

Additional literature may be added as the project develops.

---

# 28. Scientific Interpretation

The experiment is not designed to assume that sequential fine-tuning must preserve a task-order fingerprint.

Several outcomes are scientifically meaningful.

### Outcome A — Strong fingerprint

CKA and/or weight distance reliably identifies the last task.

This would support the hypothesis that sequential learning leaves measurable geometric traces of task order.

### Outcome B — Representation fingerprint but weak weight fingerprint

CKA identifies the last task but parameter distance does not.

This would suggest that the task-order information is more evident in learned representations than in raw parameter space.

### Outcome C — Weight fingerprint but weak CKA fingerprint

This would suggest that parameter-space differences contain information not captured by the selected representation comparison.

### Outcome D — No reliable fingerprint

If neither method predicts the last task above chance, this is also an important result.

It would suggest that the final representation does not reliably reveal the order under the chosen training conditions.

### Outcome E — Fingerprint depends on layer

Early layers may show weak task-order effects while later layers show strong effects, or vice versa.

This would provide evidence that task-order information is localized within particular parts of the network.

---

# 29. Important Experimental Controls

To make the conclusions meaningful, we will control for:

- identical architecture
- identical task definitions
- identical preprocessing
- consistent optimization settings
- fixed probe images
- known task orders
- multiple random seeds
- consistent checkpoint naming
- consistent evaluation procedures

We will also distinguish between:

**training differences caused by task order**

and

**differences caused by random initialization or experimental settings.**

---

# 30. Initial Local Experiment

The first executable experiment will intentionally be small.

### Configuration

```text
Dataset: CIFAR-100
Tasks: 3 selected superclasses
Model: ResNet18
Epochs: 1
Seed: 1
Order: one sequential order
```

The purpose is only to verify:

```text
dataset
   ↓
model
   ↓
training
   ↓
checkpoint
   ↓
loading
   ↓
evaluation
   ↓
feature extraction
```

Once this works, we will increase the experiment size.

---

# 31. Full Experiment

The full university-server experiment will eventually use:

```text
3 tasks
×
6 task orders
×
multiple random seeds
```

with:

```text
single-task reference models
+
sequential intermediate checkpoints
+
sequential final checkpoints
```

followed by:

```text
Weight distance
CKA
Feature drift
Forgetting
Loss barriers
Optional Jacobians
```

The exact number of epochs, seeds, probe samples, and computational settings will be finalized after the smoke test and initial pilot results.

---

# 32. Definition of Success

The project will be considered successful if the complete pipeline can:

1. Construct the three tasks reproducibly.
2. Train independent single-task ResNet18 models.
3. Train sequential models in all six task orders.
4. Save and reload all required checkpoints.
5. Measure task performance throughout sequential learning.
6. Calculate weight distances.
7. Extract internal representations.
8. Calculate CKA similarity.
9. Measure feature drift.
10. Quantify forgetting.
11. Predict the most recent task from the final checkpoint.
12. Compare prediction performance across metrics.
13. Repeat the experiment across multiple random seeds.
14. Produce reproducible tables and figures.
15. Interpret the results in the context of the cited continual-learning literature.

---

# 33. Current Status

### Planned

The complete research pipeline described above.

### Current implementation status

**Not yet implemented.**

We are starting from scratch.

### First implementation target

The first coding milestone is:

```text
CIFAR-100
    ↓
Three superclass tasks
    ↓
ResNet18
    ↓
One-epoch local training
    ↓
Checkpoint
    ↓
Evaluation
```

After this smoke test works, we will build the single-task and sequential training pipelines.

---

# 34. Guiding Principle

The project will be developed in two modes:

**Local machine:**

> Make sure everything works.

**University server:**

> Run the scientifically meaningful experiment.

We will not spend university-server compute debugging basic implementation errors.

Every major component should therefore have a small local test before being used in the full experiment.