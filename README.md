# KAN Continual Learning Framework

A modular PyTorch research framework for **domain-incremental learning (Domain-IL)** with Kolmogorov–Arnold Networks. It includes interchangeable univariate basis functions, three KAN continual-learning architecture prototypes, regularization and replay strategies, image-folder and synthetic datasets, training/evaluation utilities, checkpoints, configuration registries, and automated tests.

> **Research status:** this is a runnable research prototype, not evidence that the proposed architectures outperform established continual-learning methods. Spline locality can reduce some parameter interference, but does not guarantee retention: later layers, shared feature extractors, and overlapping feature distributions can still cause forgetting. The framework measures retention and forgetting so this can be tested.

## Included architectures

| Architecture | Registered name | Main mechanism |
|---|---|---|
| Vanilla KAN | `kan` | Learnable univariate basis expansion on each input-output edge |
| FastKAN / RBF-KAN | `fastkan` | Gaussian RBF expansion variant |
| KAC-style RBF classifier | `kac` | RBF-KAN classification model |
| Protected-Spline KAN | `protected_spline_kan` | B-spline KAN designed to pair with per-coefficient EWC |
| Adaptive-Router KAN | `adaptive_router_kan` | Domain-specific KAN expert bank plus an input-driven router |
| Hybrid-Basis MoE KAN | `hybrid_moe_kan` | Gated experts using different basis families |
| MLP baseline | `mlp` | Conventional fully connected baseline |
| CNN baseline | `cnn` | Compact CNN classifier for image-folder experiments |
| Progressive KAN | `progressive_kan` | Frozen old columns with lateral KAN connections; task-ID-aware baseline |
| Vision KAN variants | `vision_kan`, `vision_protected_spline_kan`, `vision_adaptive_router_kan`, `vision_hybrid_moe_kan` | Small CNN feature extractor followed by the selected KAN head |

The adaptive router can be trained with domain labels (`domain_labels=...`) while using its predicted router distribution at inference, where domain IDs are not supplied. Progressive KAN is included as a task-aware comparison baseline: evaluation selects the corresponding column using `task_id`, unlike domain-agnostic Domain-IL deployment. `add_domain()` appends an expert and expands the router head. The Hybrid MoE supports optional top-k routing, a load-balancing loss, and an optional expert-output diversity penalty.

## Basis families

Registered basis implementations:

- `bspline`: clamped uniform B-spline basis, evaluated with Cox–de Boor recursion.
- `rbf`: Gaussian radial basis functions with trainable centers and positive trainable widths.
- `chebyshev`: Chebyshev polynomial basis.
- `fourier`: sine/cosine Fourier features.
- `wavelet`: Mexican-hat (Ricker) wavelet basis.
- `rational`: rational polynomial basis functions.

All bases return `[batch, input_features, n_basis]`. The `KANLinear` layer contracts this tensor with edge-specific coefficients. The optional SiLU-linear residual branch provides a conventional path alongside the basis expansion. The B-spline supports are compact in the bounded input coordinate; the smooth `tanh` coordinate map keeps hidden activations in range.

## Continual-learning components

| Component | Use |
|---|---|
| Online EWC | Diagonal importance-weighted anchoring; default filter protects each KAN `coefficients` tensor element (including individual knots) |
| Synaptic Intelligence (SI) | Online path-integral parameter importance |
| Memory Aware Synapses (MAS) | Importance estimated from output sensitivity |
| Learning without Forgetting (LwF) | Frozen teacher snapshots and temperature-scaled distillation on current/replay inputs |
| Replay buffer | CPU-side reservoir sampling of input, class label, and source domain ID |
| Continual metrics | Accuracy matrix, average accuracy, forgetting, backward transfer, and first-domain retention |

EWC uses a batch-gradient diagonal Fisher approximation. This is a practical approximation, not an exact per-example Fisher calculation. Set `parameter_filter: null` to regularize all trainable parameters, including the vision feature extractor; the default `coefficients` setting protects KAN edge coefficients only.

SI and MAS are registered and can be selected from configuration. LwF can be enabled independently with `continual.lwf_weight` and `continual.lwf_temperature`; the trainer snapshots a frozen teacher after each domain and applies distillation during subsequent domains.

## Installation

Python 3.10+ and PyTorch 2.0+ are required.

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dev]"
```

For a CPU-only environment, install the appropriate CPU PyTorch build first, then run the editable install. The project does not bundle benchmark datasets or pretrained weights.

## Quick start: synthetic domain-IL

```bash
kan-cl --list-models
kan-cl --list-bases

kan-cl --config configs/synthetic_protected_spline.yaml --output outputs/protected
kan-cl --config configs/synthetic_adaptive_router.yaml --output outputs/router
kan-cl --config configs/synthetic_hybrid_moe.yaml --output outputs/hybrid_moe
```

Equivalent module invocation:

```bash
python -m kan_cl.cli --config configs/synthetic_protected_spline.yaml --output outputs/protected
```

The synthetic generator creates the same classification problem under different affine input-domain transformations. Results are intended as a fast smoke test, not a benchmark claim. A matched example comparison across MLP, vanilla KAN, FastKAN, KAC, protected KAN, adaptive routing, hybrid MoE, and progressive KAN is available with `python examples/compare_architectures.py`; a sample CPU result is included at `examples/sample_architecture_comparison.json`.

## Image benchmarks: PACS, Office-Home, DomainNet, VLCS

The built-in `image_folder` loader expects one folder per domain, then one folder per class. It can be used with PACS, Office-Home, DomainNet, VLCS, or similarly organized datasets:

```text
data/
  pacs/
    art_painting/
      dog/0001.jpg
      horse/0002.jpg
    cartoon/
      dog/0003.jpg
      horse/0004.jpg
    photo/
      ...
    sketch/
      ...
```

The loader discovers a shared class-to-index mapping across all specified domains. Example configurations are provided:

- `configs/pacs_image_folder.yaml`
- `configs/office_home_image_folder.yaml`
- `configs/domainnet_image_folder.yaml`
- `configs/vlcs_image_folder.yaml`

Run one after editing `dataset.root` to the extracted dataset directory:

```bash
kan-cl --config configs/pacs_image_folder.yaml --output outputs/pacs --device cuda
```

The vision variants use a compact trainable CNN encoder and a KAN head. The `cnn` model is the matched conventional CNN classifier baseline. They are intentionally small, dependency-free starting points, not pretrained ResNet/ViT replacements. For a strong benchmark, compare (at minimum) a matched CNN baseline, pretrained frozen/fine-tuned features, and the KAN head under identical data splits and parameter/compute budgets. Feature extraction with an external backbone can also be used by saving feature tensors and training the vector KAN models.

## Configuration

Example (shortened):

```yaml
seed: 17
device: cpu

dataset:
  type: synthetic             # synthetic | image_folder
  num_domains: 3
  samples_per_domain: 240
  input_dim: 2
  num_classes: 3
  batch_size: 48

model:
  name: protected_spline_kan
  hidden_dims: [32, 32]
  basis: bspline
  n_basis: 8
  degree: 3

optimizer:
  name: adamw                 # adam | adamw | sgd
  lr: 0.002
  weight_decay: 0.0001

loss:
  name: cross_entropy         # cross_entropy | nll | mse | bce_with_logits

training:
  epochs: 3

continual:
  ewc_strength: 0.05          # used when no `strategy` block is provided
  replay_capacity: 64
  replay_batch_size: 16
```

Alternative strategies can be selected via a `strategy` block, for example:

```yaml
continual:
  strategy:
    name: si                  # ewc | si | mas
    strength: 0.2
    damping: 0.1               # SI-specific
  replay_capacity: 64
  replay_batch_size: 12
```

For EWC, strategy-specific options include `decay` and `parameter_filter`. For MAS, use `strength`. See `configs/synthetic_si.yaml`, `configs/synthetic_mas.yaml`, and `configs/office_home_image_folder.yaml`.

## Python API

```python
import torch
from torch.utils.data import DataLoader

from kan_cl.data import make_synthetic_domains
from kan_cl.engine.build import build_model
from kan_cl.engine.trainer import run_domain_sequence

# Each dataset contains (features, labels) for one sequential domain.
domains = make_synthetic_domains(num_domains=3, samples_per_domain=256)
loaders = [DataLoader(ds, batch_size=32, shuffle=True) for ds in domains]

model = build_model({
    "name": "protected_spline_kan",
    "input_dim": 2,
    "hidden_dims": [32, 32],
    "output_dim": 3,
    "n_basis": 8,
})
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
trainer, metrics = run_domain_sequence(
    model, loaders, loaders, optimizer,
    epochs=3,
    ewc_strength=0.1,
    replay_capacity=64,
    replay_batch_size=16,
    checkpoint_dir="outputs/checkpoints",
)
print(metrics.summary())
```

## Outputs and checkpoints

Each experiment writes:

```text
outputs/<run>/
  results.json             # config, per-domain history, accuracy matrix and summary
  final_model.pt           # final model state_dict
  checkpoints/
    after_domain_0.pt       # model + optimizer + regularizer + replay state
    after_domain_1.pt
    ...
```

`ContinualTrainer.save_checkpoint()` and `load_checkpoint()` expose checkpoint persistence for custom training loops. Construct a matching model, optimizer, and selected regularizer before loading a checkpoint.

## Metrics

The accuracy matrix `A[t, d]` stores the accuracy on domain `d` after training through domain `t`. For Progressive KAN, evaluation uses the corresponding task ID, so treat its scores as task-aware baseline results rather than directly comparable task-agnostic inference. The summary reports:

- **Average accuracy:** mean accuracy across domains after the final domain.
- **Forgetting:** for each earlier domain, its best observed accuracy minus its final accuracy.
- **Backward transfer:** final accuracy on an earlier domain minus its accuracy immediately after learning that domain.
- **First-domain retention:** final accuracy on domain zero.

The diagonal and lower-triangular values are populated as domains are learned and evaluated; future-domain cells remain `NaN`.

## Tests

Run from the repository root:

```bash
pytest
```

Tests cover all six basis families and gradients, B-spline partition/local support, KAN layers, the three proposed architectures, vision wrappers, image-folder label consistency, registry construction, regularization state, replay, metrics, and synthetic sequential-training smoke tests.

## Design notes and limitations

1. **Local bases are not a complete continual-learning solution.** The output of a KAN layer still affects shared downstream parameters. The framework therefore combines basis locality with EWC/SI/MAS, explicit routing, and replay options.
2. **Adaptive routing assumes domain labels during training.** It is domain-ID-free only at inference, where its router predicts among domains that have already been registered. Open-set domain discovery is not implemented.
3. **Expert-bank growth increases memory and compute.** `add_domain()` adds a full KAN expert. Expert pruning/merging and automatic basis refinement are not yet implemented.
4. **Hybrid MoE routing is learned jointly.** Router collapse is possible; use the load-balancing term, inspect gate usage, and run routing ablations.
5. **The provided CNN is a small baseline.** Large-scale DomainNet/PACS/Office-Home research should include pretrained-backbone and matched-capacity baselines.
6. **Only classification is wired into the sequential experiment trainer.** Basis layers and KAN networks themselves are generic differentiable modules; regression, detection, and other task-specific trainers require corresponding loss/metric loops.
7. **Literature claims and novelty need independent verification.** The package implements the mechanisms described in the supplied project brief. It does not assert that any specific cited paper's results or novelty claims have been reproduced.

## Suggested research ablations

- Vanilla B-spline KAN vs protected-spline KAN with EWC strengths `0`, `0.01`, `0.1`, `1.0`.
- Per-knot EWC vs all-parameter EWC (`parameter_filter: null`).
- Adaptive router with known-domain routing vs predicted routing, and with/without replay.
- Hybrid basis set: spline-only, spline+RBF, spline+RBF+Chebyshev, and all four bases including wavelets.
- Top-1 vs top-2 routing; inspect per-domain expert gate distributions.
- Frozen vs trainable CNN backbone; report parameters, training time, and memory alongside accuracy/forgetting.
- Sequential order permutations and multiple random seeds; report mean and standard deviation.
