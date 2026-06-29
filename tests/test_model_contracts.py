import torch

from fault_experiments.metrics import binary_segmentation_metrics
from fault_experiments.models import build_model


def test_unet_and_hybrid_shape_contracts():
    x = torch.zeros(1, 1, 32, 32, 32)
    for name in ("unet3d", "dsa_hybrid"):
        model = build_model(name, base_channels=8)
        with torch.no_grad():
            y = model(x)
        assert y.shape == x.shape


def test_binary_metrics_perfect_prediction():
    label = torch.tensor([[[[[0.0, 1.0]]]]])
    logits = torch.tensor([[[[[-20.0, 20.0]]]]])
    result = binary_segmentation_metrics(logits, label, threshold=0.5)
    assert result["dice"] == 1.0
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0

