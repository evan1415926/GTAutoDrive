"""Test data loading from PNG directory."""
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
from data.loader import load_data, save_batch, load_batch


def make_frame(r, g, b):
    """Create a 180x320 RGB frame."""
    frame = np.zeros((180, 320, 3), dtype=np.uint8)
    frame[:, :] = [r, g, b]
    return frame


class TestLoader:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_data_empty_raises(self):
        import pytest
        with pytest.raises(FileNotFoundError):
            load_data(self.tmpdir)

    def test_save_load_batch_roundtrip(self):
        frames = np.random.randint(0, 255, (10, 180, 320, 3), dtype=np.uint8)
        labels = np.array([0, 0, 1, 1, 2, 2, 3, 4, 5, 6], dtype=np.int64)
        path = os.path.join(self.tmpdir, "test.npz")
        save_batch(frames, labels, path)
        loaded_frames, loaded_labels = load_batch(path)
        assert np.array_equal(frames, loaded_frames)
        assert np.array_equal(labels, loaded_labels)

    def test_load_data_from_pngs(self):
        # Create PNG files in label subdirs
        for label in ['W', 'A', 'D']:
            label_dir = Path(self.tmpdir) / label
            label_dir.mkdir()
            for i in range(3):
                frame = make_frame(i * 80, i * 40, i * 20)
                cv2.imwrite(str(label_dir / f"frame_{i:05d}.png"),
                            cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        frames, labels = load_data(self.tmpdir)
        assert len(frames) == 9
        assert len(labels) == 9
        assert frames.shape == (9, 180, 320, 3)
        assert frames.dtype == np.uint8
        # Check that all 3 W frames loaded as class 0
        assert np.all(labels[:3] == 0)
