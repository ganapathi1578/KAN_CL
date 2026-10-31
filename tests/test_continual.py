import torch
from torch.utils.data import DataLoader, TensorDataset

from kan_cl.continual.metrics import ContinualMetrics
from kan_cl.continual.regularization import MASImportance, OnlineEWC, SynapticIntelligence, lwf_distillation_loss
from kan_cl.continual.replay import ReplayBuffer
from kan_cl.models.layers import KANNetwork


def tiny_loader():
    x = torch.randn(24, 2)
    y = (x[:, 0] > x[:, 1]).long()
    return DataLoader(TensorDataset(x, y), batch_size=8, shuffle=False)


def test_ewc_importance_penalty_and_state_roundtrip():
    model = KANNetwork(2, [6], 2, n_basis=5)
    ewc = OnlineEWC(model, strength=0.5)
    data = tiny_loader()
    importance = ewc.estimate_importance(model, data)
    assert importance
    ewc.consolidate(model, importance)
    assert abs(ewc.penalty(model).item()) < 1e-10
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if "coefficients" in n).add_(0.1)
    penalty = ewc.penalty(model)
    assert penalty.item() > 0
    state = ewc.state_dict()
    restored = OnlineEWC(model)
    restored.load_state_dict(state)
    assert torch.allclose(restored.penalty(model), penalty)


def test_si_penalty_is_finite():
    model = KANNetwork(2, [6], 2, n_basis=5)
    si = SynapticIntelligence(model)
    for x, y in tiny_loader():
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        before = [p.detach().clone() for p in model.parameters()]
        torch.optim.SGD(model.parameters(), lr=0.05).step()
        si.update_after_step(model)
    si.end_task(model)
    assert torch.isfinite(si.penalty(model))


def test_mas_importance_and_penalty():
    model = KANNetwork(2, [6], 2, n_basis=5)
    mas = MASImportance(model)
    mas.estimate(model, tiny_loader())
    with torch.no_grad():
        next(model.parameters()).add_(0.1)
    assert mas.penalty(model).item() > 0


def test_lwf_distillation_is_nonnegative():
    student, teacher = torch.randn(4, 3), torch.randn(4, 3)
    assert lwf_distillation_loss(student, teacher).item() >= 0


def test_replay_buffer_capacity_and_sampling():
    replay = ReplayBuffer(capacity=10, seed=4)
    replay.add_batch(torch.randn(15, 2), torch.arange(15) % 2, domain_id=1)
    assert len(replay) == 10
    x, y, d = replay.sample(4)
    assert x.shape == (4, 2) and y.shape == (4,) and (d == 1).all()
    state = replay.state_dict()
    restored = ReplayBuffer(1)
    restored.load_state_dict(state)
    assert len(restored) == 10


def test_continual_metric_calculation():
    metrics = ContinualMetrics(3)
    metrics.update(0, 0, 0.8)
    metrics.update(1, 0, 0.7)
    metrics.update(1, 1, 0.75)
    metrics.update(2, 0, 0.6)
    metrics.update(2, 1, 0.7)
    metrics.update(2, 2, 0.9)
    summary = metrics.summary()
    assert abs(summary["average_accuracy"] - (0.6 + 0.7 + 0.9) / 3) < 1e-8
    assert abs(summary["first_domain_retention"] - 0.6) < 1e-8
    assert abs(summary["forgetting_by_domain"][0] - 0.2) < 1e-8
