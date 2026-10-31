from pathlib import Path

from kan_cl.engine.experiment import load_config, run_experiment


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "synthetic_protected_spline.yaml")
    result = run_experiment(config, root / "outputs" / "protected_spline")
    print(result["metrics"])
