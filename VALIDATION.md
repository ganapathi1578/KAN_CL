# Validation report

Validation was run in the provided CPU-only runtime using PyTorch 2.10.0+cpu.

## Automated tests

Command:

```bash
python -m pytest -q
```

Result: **35 tests passed**.

Coverage includes:

- Six basis functions, finite outputs, and gradient flow.
- B-spline partition of unity and compact local support.
- KAN layers/networks and trainable RBF centers/widths.
- Protected-Spline, Adaptive-Router, Hybrid-Basis MoE, Progressive KAN, FastKAN, KAC-style RBF, Vision KAN, and CNN baseline forward/training paths.
- Dynamic router/progressive expansion and optimizer parameter synchronization, including EWC/SI/MAS state resizing when a router head grows.
- EWC, SI, MAS, LwF, replay state, and checkpoint round-trips.
- Continual accuracy/forgetting metrics.
- Shared class-label mapping for image-folder domains.
- End-to-end synthetic sequential training.

## Smoke runs

The following configs completed end-to-end on CPU and emitted `results.json`, final model weights, and per-domain checkpoints:

- `synthetic_protected_spline.yaml`
- `synthetic_adaptive_router.yaml`
- `synthetic_hybrid_moe.yaml`
- `synthetic_progressive.yaml`
- `synthetic_si.yaml`
- `synthetic_mas.yaml`
- `synthetic_lwf.yaml`

A separate generated image-folder fixture with two domains, two classes, and 32×32 RGB images completed a full CNN-to-Protected-Spline-KAN run, including EWC, replay, and LwF. The conventional CNN image baseline also completed the same two-domain image-folder smoke path.

The matched architecture comparison script completed for MLP, vanilla KAN, FastKAN, KAC-style RBF, Protected-Spline KAN, Adaptive-Router KAN, Hybrid-Basis MoE KAN, and Progressive KAN. Its small synthetic output is included as `examples/sample_architecture_comparison.json`.

These runs establish that the package executes end-to-end. The small synthetic metrics are not representative benchmark results and should not be interpreted as evidence of superiority; the deliberately small fixture can produce unstable accuracy/forgetting estimates.

## Installation check

Editable installation succeeded with:

```bash
pip install -e . --no-build-isolation --no-deps
```

The default isolated build step may need internet access to fetch build requirements in a fresh environment. Runtime dependencies are listed in `requirements.txt` and `pyproject.toml`.

## Not verified here

- CUDA training/performance (the validation runtime exposed CPU-only PyTorch).
- Full PACS, Office-Home, VLCS, DomainNet, Split-CIFAR, or other benchmark runs; datasets are not bundled.
- Statistical significance or publication-quality comparisons across multiple seeds.
