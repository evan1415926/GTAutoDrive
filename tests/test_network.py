"""Test model forward pass shapes."""
import torch
from model.network import create_model


class TestNetwork:
    def test_forward_shape(self):
        model = create_model(num_classes=7, dropout=0.5)
        model.eval()
        # Simulate 320x180 RGB input
        x = torch.randn(4, 3, 180, 320).mul(255)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 7)

    def test_forward_shape_single(self):
        model = create_model(num_classes=7, dropout=0.5)
        model.eval()
        x = torch.randn(1, 3, 180, 320).mul(255)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1, 7)

    def test_different_num_classes(self):
        model = create_model(num_classes=3, dropout=0.3)
        x = torch.randn(2, 3, 180, 320).mul(255)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 3)
