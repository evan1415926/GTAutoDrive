"""End-to-end smoke test: model -> inference -> action on synthetic frame."""
import numpy as np
import torch
from model.network import create_model
from config.settings import IDX_TO_LABEL, LABEL_TO_KEYS


class TestIntegration:
    def test_model_inference_roundtrip(self):
        """Model takes synthetic frame -> outputs a valid action label."""
        model = create_model(num_classes=7)
        model.eval()

        # Random frame
        frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
        tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()

        with torch.no_grad():
            logits = model(tensor)

        probs = torch.softmax(logits, dim=1).numpy().flatten()
        action_idx = int(np.argmax(probs))
        action = IDX_TO_LABEL[action_idx]

        # Action must be valid
        assert action in LABEL_TO_KEYS
        # Probabilities sum to 1
        assert abs(probs.sum() - 1.0) < 0.01

    def test_ema_stability(self):
        """EMA smoothing should not crash on repeated calls."""
        alpha = 0.4
        ema_probs = np.ones(7) / 7.0
        model = create_model(num_classes=7)
        model.eval()

        for _ in range(100):
            frame = np.random.randint(0, 255, (180, 320, 3), dtype=np.uint8)
            tensor = torch.from_numpy(frame).permute(
                2, 0, 1).unsqueeze(0).float()
            with torch.no_grad():
                logits = model(tensor)
            probs = torch.softmax(logits, dim=1).numpy().flatten()
            ema_probs = alpha * probs + (1 - alpha) * ema_probs

        assert abs(ema_probs.sum() - 1.0) < 0.01
        assert np.all(ema_probs >= -1e-6) and np.all(ema_probs <= 1 + 1e-6)

    def test_all_labels_have_key_mapping(self):
        """Every label must map to a valid key state."""
        from config.settings import LABELS, LABEL_TO_KEYS
        for label in LABELS:
            keys = LABEL_TO_KEYS[label]
            assert 'w' in keys and 'a' in keys and 's' in keys and 'd' in keys
