# -*- coding: utf-8 -*-
import cv2
import numpy as np

import brica
from .utils import load_image
from .constants import MODEL_PATHS

"""
This is a sample implemention of PFC (Prefrontal cortex) module.
You can change this as you like.
"""

class Phase(object):
    INIT = -1  # Initial phase
    START = 0  # Start phase while finding red cross cursor
    TARGET = 1 # Target finding phsae


class CursorFindAccumulator(object):
    def __init__(self, decay_rate=0.9):
        # Accumulated likelilood
        self.decay_rate = decay_rate
        self.likelihood = 0.0

        self.cursor_template = load_image("data/debug_cursor_template_w.png")

    def accumulate(self, value):
        self.likelihood += value
        self.likelihood = np.clip(self.likelihood, 0.0, 1.0)

    def reset(self):
        self.likelihood = 0.0

    def process(self, retina_image):
        match = cv2.matchTemplate(retina_image, self.cursor_template,
                                  cv2.TM_CCOEFF_NORMED)
        match_rate = np.max(match)
        self.accumulate(match_rate * 0.1)

    def post_process(self):
        # Decay likelihood
        self.likelihood *= self.decay_rate

class Accumulator:
    def __init__(self, decay_rate=0.9):
        self.decay_rate = decay_rate
        self.value = 0.0

    def accumulate(self, value):
        self.value *= self.decay_rate
        self.value += value

    def reset(self):
        self.value = 0.0

    def get_value(self):
        return self.value

class PFC(object):
    def __init__(self):
        self.timing = brica.Timing(3, 1, 0)

        self.cursor_find_accmulator = CursorFindAccumulator()
        self.vae_error_accumulators = {}
        for name in MODEL_PATHS.keys():
            self.vae_error_accumulators[name]= Accumulator()

        self.phase = Phase.INIT
        self.prev_phase = self.phase
        self.last_current_task = None

    def __call__(self, inputs):
        if 'from_vc' not in inputs:
            raise Exception('PFC did not recieve from VC')
        if 'from_fef' not in inputs:
            raise Exception('PFC did not recieve from FEF')
        if 'from_bg' not in inputs:
            raise Exception('PFC did not recieve from BG')
        if 'from_hp' not in inputs:
            raise Exception('PFC did not recieve from HP')

        # Image from Visual cortex module.
        retina_image, pixel_errors, top_errors = inputs['from_vc']
        # Allocentrix map image from hippocampal formatin module.
        map_image = inputs['from_hp']

        # This is a very sample implementation of phase detection.
        # You should change here as you like.
        self.cursor_find_accmulator.process(retina_image)
        self.cursor_find_accmulator.post_process()
        bg_message = 0
        bg_findcursor = 0

        if self.phase == Phase.INIT:
            if self.cursor_find_accmulator.likelihood > 0.7:
                self.phase = Phase.START
                bg_message = 1
                bg_findcursor = 1
        elif self.phase == Phase.START:
            if self.cursor_find_accmulator.likelihood < 0.4:
                self.phase = Phase.TARGET
                bg_message = 1
            else:
                bg_findcursor = 1
        else:
            if self.cursor_find_accmulator.likelihood > 0.6:
                self.phase = Phase.START

        # accumulate reconstruction error
        errors = []
        for name, pixel_error in pixel_errors.items():
            errors.append(pixel_error.mean(-1).mean(-1).mean(-1))
        current_task = list(pixel_errors.keys())[np.argmin(errors)]
        self.last_current_task = current_task

        # TODO: update fef_message signal
        if self.phase == Phase.INIT or self.phase == Phase.START:
            # TODO: 領野をまたいだ共通phaseをどう定義するか？
            # original 0
            fef_message = 0
        else:
            fef_message = 1

        return dict(to_fef=(fef_message, current_task),
                    to_bg=(bg_message, bg_findcursor, current_task))
