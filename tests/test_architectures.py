import torch
import torch.nn.functional as F

from kan_cl.models.architectures import AdaptiveRouterKAN, HybridBasisMoEKAN, ProtectedSplineKAN


def test_protected_spline_architecture():
    model = ProtectedSplineKAN(3, [8], 2, n_basis=6)
    assert model(torch.randn(4, 3)).shape == (4, 2)
    assert all("coefficients" in name for name, _ in model.named_parameters() if "coefficients" in name)


def test_adaptive_router_supervised_and_inferred_routing():
    model = AdaptiveRouterKAN(3, [8], 2, num_domains=2, n_basis=6)
    x = torch.randn(5, 3)
    y, aux = model(x, domain_labels=torch.tensor([0, 0, 1, 1, 0]), return_aux=True)
    assert y.shape == (5, 2)
    assert aux["router_logits"].shape == (5, 2)
    assert torch.allclose(aux["gates"].sum(-1), torch.ones(5))
    inferred = model(x)
    assert inferred.shape == (5, 2)
    loss = F.cross_entropy(inferred, torch.randint(0, 2, (5,)))
    loss.backward()
    assert model.router[-1].weight.grad is not None


def test_adaptive_router_can_expand_domain():
    model = AdaptiveRouterKAN(2, [6], 3, num_domains=1, n_basis=5)
    new_id = model.add_domain(copy_from=0)
    assert new_id == 1
    assert model.num_domains == 2
    out = model(torch.randn(4, 2), domain_labels=torch.ones(4, dtype=torch.long))
    assert out.shape == (4, 3)


def test_hybrid_moe_and_auxiliary_losses():
    model = HybridBasisMoEKAN(3, [8], 2, expert_bases=("bspline", "rbf", "chebyshev"),
                              n_basis=6, top_k=2)
    output, aux = model(torch.randn(7, 3), return_aux=True)
    assert output.shape == (7, 2)
    assert (aux["gates"] > 0).sum(-1).eq(2).all()
    assert torch.isfinite(model.load_balance_loss(aux["gates"]))
    assert torch.isfinite(model.diversity_loss(aux["expert_outputs"]))


def test_vision_wrappers_for_all_novel_heads():
    from kan_cl.models.vision import VisionKAN

    image = torch.randn(2, 3, 32, 32)
    protected = VisionKAN(output_dim=3, head_type="protected_spline_kan",
                          feature_dim=16, hidden_dims=[8], n_basis=5)
    assert protected(image).shape == (2, 3)
    router = VisionKAN(output_dim=3, head_type="adaptive_router_kan", num_domains=2,
                       feature_dim=16, hidden_dims=[8], n_basis=5)
    prediction, aux = router(image, domain_labels=torch.tensor([0, 1]), return_aux=True)
    assert prediction.shape == (2, 3)
    assert aux["router_logits"].shape == (2, 2)
    moe = VisionKAN(output_dim=3, head_type="hybrid_moe_kan",
                    expert_bases=("bspline", "rbf"), feature_dim=16,
                    hidden_dims=[8], n_basis=5, top_k=1)
    assert moe(image).shape == (2, 3)


def test_progressive_kan_adds_frozen_columns_with_lateral_paths():
    from kan_cl.models.architectures import ProgressiveKAN

    model = ProgressiveKAN(3, [8, 6], 2, num_tasks=1, n_basis=5)
    x = torch.randn(4, 3)
    first = model(x, task_id=0)
    model.add_domain()
    second = model(x, task_id=1)
    assert first.shape == second.shape == (4, 2)
    assert all(not p.requires_grad for p in model.columns[0].parameters())
    assert any(p.requires_grad for p in model.columns[1].parameters())


def test_fastkan_and_kac_registry_variants():
    from kan_cl.engine.build import build_model

    for name in ("fastkan", "kac"):
        model = build_model({"name": name, "input_dim": 3,
                             "hidden_dims": [6], "output_dim": 2, "n_basis": 5})
        assert model(torch.randn(4, 3)).shape == (4, 2)
        assert model.basis_name == "rbf"


def test_cnn_image_classifier_baseline():
    from kan_cl.engine.build import build_model

    model = build_model({"name": "cnn", "output_dim": 4, "feature_dim": 16})
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 4)
