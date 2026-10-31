from pathlib import Path

import numpy as np
import torch
from PIL import Image

from kan_cl.data.folder import DomainFolderDataset, discover_shared_classes
from kan_cl.engine.build import build_model
from kan_cl.engine.components import DATASETS, LOSSES, OPTIMIZERS, STRATEGIES, build_loss, build_optimizer


def test_image_folder_dataset_shared_class_labels(tmp_path: Path):
    for domain, names in [("photo", ["cat", "dog"]), ("sketch", ["dog", "cat"])]:
        for class_name in names:
            folder = tmp_path / domain / class_name
            folder.mkdir(parents=True)
            array = np.full((20, 20, 3), 128, dtype=np.uint8)
            Image.fromarray(array).save(folder / "sample.png")
    class_map = discover_shared_classes(tmp_path, ["photo", "sketch"])
    photo = DomainFolderDataset(tmp_path, "photo", class_map, image_size=16)
    sketch = DomainFolderDataset(tmp_path, "sketch", class_map, image_size=16)
    assert len(photo) == len(sketch) == 2
    assert photo[0][0].shape == (3, 16, 16)
    assert photo.class_to_idx == sketch.class_to_idx
    assert {label for _, label in photo.samples} == {0, 1}
    assert {label for _, label in sketch.samples} == {0, 1}


def test_registries_expose_components_and_builders():
    assert "synthetic" in DATASETS
    assert "adamw" in OPTIMIZERS
    assert "cross_entropy" in LOSSES
    assert "ewc" in STRATEGIES
    model = build_model({"name": "vision_protected_spline_kan", "output_dim": 2,
                         "feature_dim": 8, "hidden_dims": [4], "n_basis": 5})
    assert model(torch.randn(2, 3, 32, 32)).shape == (2, 2)
    loss = build_loss("cross_entropy")
    optimizer = build_optimizer("adam", model.parameters(), lr=1e-3)
    assert loss(torch.randn(2, 2), torch.tensor([0, 1])).item() > 0
    assert optimizer is not None
