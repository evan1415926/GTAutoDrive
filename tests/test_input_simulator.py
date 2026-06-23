"""Tests for InputSimulator — state tracking only (no real key events)."""
import pytest
from input.input_simulator import InputSimulator
from config.settings import KeysConfig


class TestInputSimulator:
    def setup_method(self):
        self.sim = InputSimulator(KeysConfig())

    def test_initial_state_all_released(self):
        assert self.sim._state == {'w': False, 'a': False, 's': False,
                                   'd': False}

    def test_apply_w_sets_w_only(self):
        # Override _ctrl to avoid real key events
        pressed = []
        released = []

        class FakeCtrl:
            def press(self, k): pressed.append(k)
            def release(self, k): released.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim.apply('W')
        assert self.sim._state['w'] is True
        assert self.sim._state['a'] is False
        assert self.sim._state['s'] is False
        assert self.sim._state['d'] is False
        assert 'w' in pressed

    def test_apply_wa_sets_w_and_a(self):
        pressed = []
        released = []

        class FakeCtrl:
            def press(self, k): pressed.append(k)
            def release(self, k): released.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim.apply('WA')
        assert self.sim._state['w'] is True
        assert self.sim._state['a'] is True
        assert self.sim._state['s'] is False
        assert self.sim._state['d'] is False

    def test_transition_w_to_wa(self):
        """Going from W to WA should only press 'a', not re-press 'w'."""
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim._state = {'w': True, 'a': False, 's': False, 'd': False}
        self.sim.apply('WA')
        assert self.sim._state['a'] is True
        assert len(releases) == 0  # no releases

    def test_transition_wd_to_s(self):
        """Brake releases everything, presses 's'."""
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim._state = {'w': True, 'a': False, 's': False, 'd': True}
        self.sim.apply('S')
        assert 'w' in releases or self.sim._state['w'] is False
        assert 'd' in releases or self.sim._state['d'] is False

    def test_apply_none_releases_all(self):
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim._state = {'w': True, 'a': False, 's': False, 'd': True}
        self.sim.apply('NONE')
        assert self.sim._state == {'w': False, 'a': False, 's': False,
                                   'd': False}

    def test_release_all(self):
        self.sim._state = {'w': True, 'a': True, 's': False, 'd': False}
        releases = []

        class FakeCtrl:
            def press(self, k): pass
            def release(self, k): releases.append(k)

        self.sim._ctrl = FakeCtrl()
        self.sim.release_all()
        assert self.sim._state == {'w': False, 'a': False, 's': False,
                                   'd': False}
        assert 'w' in releases
        assert 'a' in releases
