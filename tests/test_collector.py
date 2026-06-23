"""Test DataCollector file output."""
import tempfile
import os
from pathlib import Path
import numpy as np
import cv2
from data.collector import DataCollector
from config.settings import DataConfig


def make_frame():
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


class TestDataCollector:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        cfg = DataConfig(save_dir=self.tmpdir, frame_width=320,
                         frame_height=180, fps=10)
        self.coll = DataCollector(cfg)

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_not_recording_by_default(self):
        assert not self.coll.is_recording

    def test_start_stop(self):
        self.coll.start()
        assert self.coll.is_recording
        self.coll.stop()
        assert not self.coll.is_recording

    def test_capture_creates_file(self):
        self.coll.start()
        self.coll.capture(make_frame(), w=True, a=False, s=False, d=False)
        self.coll.stop()
        # Should have created data/recordings/W/frame_00000.png
        expected = Path(self.tmpdir) / "W" / "frame_00000.png"
        assert expected.exists()
        # Verify dimensions
        img = cv2.imread(str(expected))
        assert img.shape == (180, 320, 3)

    def test_capture_multiple_labels(self):
        self.coll.start()
        self.coll.capture(make_frame(), w=True, a=False, s=False, d=False)  # W
        self.coll.capture(make_frame(), w=True, a=True, s=False, d=False)   # WA
        self.coll.capture(make_frame(), w=False, a=False, s=True, d=False)  # S
        self.coll.stop()
        assert (Path(self.tmpdir) / "W" / "frame_00000.png").exists()
        assert (Path(self.tmpdir) / "WA" / "frame_00001.png").exists()
        assert (Path(self.tmpdir) / "S" / "frame_00002.png").exists()
        assert self.coll.frame_count == 3

    def test_toggle_mode(self):
        assert self.coll.mode == 'train'
        self.coll.toggle_mode()
        assert self.coll.mode == 'recovery'
        self.coll.toggle_mode()
        assert self.coll.mode == 'train'
