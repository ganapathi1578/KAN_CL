import torch
from torch.utils.data import DataLoader

from kan_cl.data.synthetic import make_synthetic_domains
from kan_cl.engine.build import build_model
from kan_cl.engine.trainer import run_domain_sequence


def test_synthetic_sequential_training_end_to_end(tmp_path):
    domains = make_synthetic_domains(num_domains=2, samples_per_domain=48, num_classes=2)
    loaders = [DataLoader(ds, batch_size=16, shuffle=True) for ds in domains]
    model = build_model({"name": "protected_spline_kan", "input_dim": 2,
                         "hidden_dims": [8], "output_dim": 2, "n_basis": 5})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    trainer, metrics = run_domain_sequence(model, loaders, loaders, optimizer,
                                           epochs=1, ewc_strength=0.01,
                                           replay_capacity=12, replay_batch_size=4,
                                           checkpoint_dir=tmp_path)
    assert len(trainer.history) == 2
    assert metrics.matrix.shape == (2, 2)
    assert (tmp_path / "after_domain_0.pt").exists()
    assert (tmp_path / "after_domain_1.pt").exists()


def test_adaptive_router_and_moe_train_through_engine():
    domains = make_synthetic_domains(num_domains=2, samples_per_domain=32, num_classes=2)
    loaders = [DataLoader(ds, batch_size=16, shuffle=False) for ds in domains]
    for model_cfg in [
        {"name": "adaptive_router_kan", "input_dim": 2, "hidden_dims": [6],
         "output_dim": 2, "num_domains": 2, "n_basis": 5},
        {"name": "hybrid_moe_kan", "input_dim": 2, "hidden_dims": [6],
         "output_dim": 2, "expert_bases": ["bspline", "rbf"], "n_basis": 5},
    ]:
        model = build_model(model_cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer, metrics = run_domain_sequence(model, loaders, loaders, optimizer, epochs=1)
        assert len(trainer.history) == 2
        assert torch.isfinite(metrics.matrix[1]).all()


def test_si_and_mas_can_be_selected_as_training_strategies(tmp_path):
    from kan_cl.engine.components import build_strategy

    domains = make_synthetic_domains(num_domains=2, samples_per_domain=24, num_classes=2)
    loaders = [DataLoader(ds, batch_size=12, shuffle=False) for ds in domains]
    for name, kwargs in [("si", {"strength": 0.1}), ("mas", {"strength": 0.1})]:
        model = build_model({"name": "kan", "input_dim": 2, "hidden_dims": [5],
                             "output_dim": 2, "n_basis": 5})
        strategy = build_strategy(name, model=model, **kwargs)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        trainer, _ = run_domain_sequence(model, loaders, loaders, optimizer, epochs=1,
                                         regularizer=strategy)
        path = tmp_path / f"{name}.pt"
        trainer.save_checkpoint(path)
        restored_model = build_model({"name": "kan", "input_dim": 2, "hidden_dims": [5],
                                      "output_dim": 2, "n_basis": 5})
        restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.001)
        restored_strategy = build_strategy(name, model=restored_model, **kwargs)
        from kan_cl.engine.trainer import ContinualTrainer
        restored = ContinualTrainer(restored_model, restored_optimizer, regularizer=restored_strategy)
        restored.load_checkpoint(path)
        assert len(restored.history) == 2


def test_dynamic_router_domain_expansion_keeps_optimizer_in_sync():
    domains = make_synthetic_domains(num_domains=2, samples_per_domain=24, num_classes=2)
    loaders = [DataLoader(ds, batch_size=12, shuffle=False) for ds in domains]
    model = build_model({"name": "adaptive_router_kan", "input_dim": 2,
                         "hidden_dims": [5], "output_dim": 2,
                         "num_domains": 1, "n_basis": 5})
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer, _ = run_domain_sequence(model, loaders, loaders, optimizer, epochs=1)
    assert model.num_domains == 2
    live_ids = {id(parameter) for parameter in model.parameters()}
    optimizer_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert live_ids == optimizer_ids


def test_progressive_kan_runs_task_aware_baseline():
    domains = make_synthetic_domains(num_domains=2, samples_per_domain=24, num_classes=2)
    loaders = [DataLoader(ds, batch_size=12, shuffle=False) for ds in domains]
    model = build_model({"name": "progressive_kan", "input_dim": 2,
                         "hidden_dims": [6, 6], "output_dim": 2,
                         "num_tasks": 1, "n_basis": 5})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer, metrics = run_domain_sequence(model, loaders, loaders, optimizer, epochs=1)
    assert model.num_tasks == 2
    assert len(trainer.history) == 2
    assert torch.isfinite(metrics.matrix[1]).all()


def test_lwf_teacher_snapshot_and_checkpoint_roundtrip(tmp_path):
    domains = make_synthetic_domains(num_domains=2, samples_per_domain=24, num_classes=2)
    loaders = [DataLoader(ds, batch_size=12, shuffle=False) for ds in domains]
    model = build_model({"name": "kan", "input_dim": 2, "hidden_dims": [5],
                         "output_dim": 2, "n_basis": 5})
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    trainer, _ = run_domain_sequence(model, loaders, loaders, optimizer, epochs=1,
                                     lwf_weight=0.2, checkpoint_dir=tmp_path)
    assert trainer.teacher is not None
    assert all(not p.requires_grad for p in trainer.teacher.parameters())
    checkpoint = tmp_path / "after_domain_1.pt"
    new_model = build_model({"name": "kan", "input_dim": 2, "hidden_dims": [5],
                             "output_dim": 2, "n_basis": 5})
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    from kan_cl.engine.trainer import ContinualTrainer
    restored = ContinualTrainer(new_model, new_optimizer, lwf_weight=0.2)
    restored.load_checkpoint(checkpoint)
    assert restored.teacher is not None


def test_classification_loss_registry_variants_run():
    from kan_cl.engine.components import build_loss
    from kan_cl.engine.trainer import ContinualTrainer

    domains = make_synthetic_domains(num_domains=1, samples_per_domain=12, num_classes=3)
    loader = DataLoader(domains[0], batch_size=6)
    for loss_name in ("cross_entropy", "nll", "mse", "bce_with_logits"):
        model = build_model({"name": "kan", "input_dim": 2, "hidden_dims": [5],
                             "output_dim": 3, "n_basis": 5})
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = ContinualTrainer(model, optimizer, loss_fn=build_loss(loss_name))
        result = trainer.train_domain(0, loader, epochs=1)
        assert torch.isfinite(torch.tensor(result["loss"]))


def test_expandable_router_works_with_importance_regularizers():
    from kan_cl.continual.regularization import MASImportance, OnlineEWC, SynapticIntelligence

    domains = make_synthetic_domains(num_domains=2, samples_per_domain=16, num_classes=2)
    loaders = [DataLoader(ds, batch_size=8, shuffle=False) for ds in domains]
    strategies = [
        OnlineEWC,
        SynapticIntelligence,
        MASImportance,
    ]
    for strategy_type in strategies:
        model = build_model({"name": "adaptive_router_kan", "input_dim": 2,
                             "hidden_dims": [4], "output_dim": 2,
                             "num_domains": 1, "n_basis": 4,
                             "router_hidden_dim": 4})
        if strategy_type is OnlineEWC:
            strategy = strategy_type(model, strength=0.01, parameter_filter=None)
        else:
            strategy = strategy_type(model, strength=0.01)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer, metrics = run_domain_sequence(
            model, loaders, loaders, optimizer, epochs=1, regularizer=strategy
        )
        assert model.num_domains == 2
        assert len(trainer.history) == 2
        assert torch.isfinite(metrics.matrix[1]).all()
        penalty = strategy.penalty(model)
        assert torch.isfinite(penalty)
