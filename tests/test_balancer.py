"""Test class balancing."""
import numpy as np
from data.balancer import balance_classes


class TestBalancer:
    def test_balance_reduces_to_min_class(self):
        # 10 W, 5 WA, 2 WD, 1 A, 1 D, 1 S, 1 NONE
        frames = np.zeros((21, 180, 320, 3), dtype=np.uint8)
        labels = np.array(
            [0] * 10 +   # W
            [1] * 5 +    # WA
            [2] * 2 +    # WD
            [3] * 1 +    # A
            [4] * 1 +    # D
            [5] * 1 +    # S
            [6] * 1,     # NONE
            dtype=np.int64
        )
        f_bal, l_bal = balance_classes(frames, labels)
        # Min class = 1, 7 classes
        assert len(f_bal) == 7
        # Each class should appear exactly once
        unique, counts = np.unique(l_bal, return_counts=True)
        assert len(unique) == 7
        assert np.all(counts == 1)

    def test_balance_preserves_shape(self):
        frames = np.zeros((14, 180, 320, 3), dtype=np.uint8)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6],
                          dtype=np.int64)
        f_bal, l_bal = balance_classes(frames, labels)
        # Min = 2, 7 classes = 14 total
        assert len(f_bal) == 14
        assert f_bal.shape == (14, 180, 320, 3)
