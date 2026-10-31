# Architecture and training flow

## Model family overview

### Protected-Spline KAN

```mermaid
flowchart TD
    X[Feature vector x] --> L1[KAN layer: B-spline edge functions]
    L1 --> H[Hidden representation]
    H --> L2[KAN layer: B-spline edge functions]
    L2 --> Y[Class logits]
    O[Per-coefficient Fisher importance Ω] --> R[EWC anchoring penalty]
    W[Saved coefficient values θ*] --> R
    L2 -. coefficients .-> R
    R --> LOSS[Task loss + λ EWC]
```

Every KAN edge has a trainable coefficient vector over local basis functions. The default `OnlineEWC` filter selects parameter names containing `coefficients`, which means the penalty is element-wise across edge coefficients/knot values. The Fisher estimate is a minibatch-gradient approximation.

### Adaptive-Router KAN

```mermaid
flowchart TD
    X[Input features] --> ROUTER[Domain router]
    ROUTER --> P[Predicted domain probabilities]
    P --> MIX[Weighted expert output]
    X --> E0[Domain KAN expert 0]
    X --> E1[Domain KAN expert 1]
    X --> ED[Domain KAN expert D]
    E0 --> MIX
    E1 --> MIX
    ED --> MIX
    MIX --> Y[Class logits]
    LABEL[Known domain label during training] --> G[One-hot training gate]
    G --> E0
    G --> E1
    G --> ED
```

During training, the known domain ID selects the corresponding expert while an auxiliary cross-entropy objective trains the router. During inference, the expert outputs are mixed using the router's predicted probabilities. This is domain-ID-free at test time only for domains already represented by the router.

### Hybrid-Basis MoE KAN

```mermaid
flowchart TD
    X[Input features] --> G[Learned gating network]
    X --> S[ B-spline KAN expert ]
    X --> R[ RBF KAN expert ]
    X --> C[ Chebyshev KAN expert ]
    X --> W[ Wavelet KAN expert ]
    G --> MIX[Top-k or dense weighted sum]
    S --> MIX
    R --> MIX
    C --> MIX
    W --> MIX
    MIX --> Y[Class logits]
```

Use `top_k` to make routing sparse. The load-balancing term discourages expert collapse. The optional diversity term discourages highly correlated expert outputs; tune it cautiously because forced diversity can also hurt task performance.

### Progressive KAN baseline

```mermaid
flowchart LR
    X[Input] --> C0[Column 0]
    X --> C1[Column 1]
    C0 -->|frozen lateral features| C1
    X --> C2[Column 2]
    C0 -->|frozen lateral features| C2
    C1 -->|frozen lateral features| C2
    C2 --> Y[Task-2 logits]
```

Earlier columns are frozen when a new task/domain column is appended. This is a task-aware comparison baseline: provide `task_id` at evaluation. It is not a domain-agnostic method.

## Sequential training loop

```mermaid
flowchart TD
    D0[Domain 0 data] --> T0[Train current model / expert / column]
    T0 --> C0[Consolidate EWC, SI, or MAS]
    C0 --> E0[Evaluate domains seen so far]
    E0 --> B0[Update replay buffer and LwF teacher]
    B0 --> D1[Next domain data]
    D1 --> T1[Train next domain]
    T1 --> C1[Consolidate importance]
    C1 --> E1[Re-evaluate all seen domains]
    E1 --> B1[Update replay buffer and teacher]
    B1 --> ...[Continue until all domains are processed]
```

At each stage, the evaluator records `A[t, d]`, the score on domain `d` after training through domain `t`. The final report derives average accuracy, forgetting, backward transfer, and first-domain retention from this matrix.

## Core implementation boundaries

- `models/basis.py`: registered basis implementations.
- `models/layers.py`: edge-wise `KANLinear`, sequential `KANNetwork`, MLP baseline.
- `models/architectures.py`: protected, adaptive-router, hybrid-MoE, progressive, FastKAN, and KAC-style models.
- `models/vision.py`: CNN feature encoder, KAN vision wrappers, conventional CNN baseline.
- `continual/regularization.py`: EWC, SI, MAS, and LwF loss.
- `continual/replay.py`: reservoir replay memory including source domain IDs.
- `continual/metrics.py`: accuracy matrix and continual-learning metrics.
- `data/`: synthetic domain generator and generic image-folder reader.
- `engine/`: registries, builders, experiment runner, trainer, and checkpointing.
