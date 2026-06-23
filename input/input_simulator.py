"""Keyboard input simulation with state tracking via pynput."""
from pynput.keyboard import Controller, Key
from config.settings import LABEL_TO_KEYS, KeysConfig


class InputSimulator:
    def __init__(self, config: KeysConfig):
        self._key_map = {
            'w': config.throttle,
            'a': config.steer_left,
            's': config.brake,
            'd': config.steer_right,
        }
        self._ctrl = Controller()
        # Track which keys are currently pressed to avoid redundant events
        self._state = {'w': False, 'a': False, 's': False, 'd': False}

    def apply(self, label: str):
        target = LABEL_TO_KEYS.get(label, LABEL_TO_KEYS['NONE'])
        for key_char in ('w', 'a', 's', 'd'):
            desired = target[key_char]
            current = self._state[key_char]
            if desired and not current:
                self._ctrl.press(self._key_map[key_char])
                self._state[key_char] = True
            elif not desired and current:
                self._ctrl.release(self._key_map[key_char])
                self._state[key_char] = False

    def release_all(self):
        for key_char in ('w', 'a', 's', 'd'):
            if self._state[key_char]:
                self._ctrl.release(self._key_map[key_char])
                self._state[key_char] = False
