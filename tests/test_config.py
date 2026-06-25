"""Test shared constants and config."""
import pytest
from config.settings import (LABELS, LABEL_TO_IDX, IDX_TO_LABEL,
                              LABEL_TO_KEYS, keys_to_label, AppConfig)


def test_labels_length():
    assert len(LABELS) == 7


def test_label_idx_roundtrip():
    for label in LABELS:
        assert IDX_TO_LABEL[LABEL_TO_IDX[label]] == label


def test_keys_to_label_w():
    assert keys_to_label(w=True, a=False, s=False, d=False) == 'W'


def test_keys_to_label_wa():
    assert keys_to_label(w=True, a=True, s=False, d=False) == 'WA'


def test_keys_to_label_wd():
    assert keys_to_label(w=True, a=False, s=False, d=True) == 'WD'


def test_keys_to_label_a():
    assert keys_to_label(w=False, a=True, s=False, d=False) == 'A'


def test_keys_to_label_d():
    assert keys_to_label(w=False, a=False, s=False, d=True) == 'D'


def test_keys_to_label_s():
    assert keys_to_label(w=False, a=False, s=True, d=False) == 'S'
    # Brake takes priority over everything
    assert keys_to_label(w=True, a=True, s=True, d=True) == 'S'


def test_keys_to_label_none():
    assert keys_to_label(w=False, a=False, s=False, d=False) == 'NONE'


def test_label_to_keys_every_label_covered():
    for label in LABELS:
        assert label in LABEL_TO_KEYS, f"Missing key mapping for {label}"


def test_appconfig_defaults():
    cfg = AppConfig()
    assert cfg.model.num_classes == 7
    assert cfg.train.batch_size == 64
    assert cfg.inference.ema_alpha == 0.15
    assert cfg.inference.confidence_threshold == 0.3
    assert cfg.keys.panic_key == 'f8'
