# -*- coding: utf-8 -*-
import cv2
import numpy as np

import brica
from .utils import load_image
from .utils import softmax
from .constants import MODEL_PATHS
from collections import OrderedDict

"""
This is a sample implemention of PFC (Prefrontal cortex) module.
You can change this as you like.
"""

class Phase(object):
    INIT = -1  # Initial phase
    START = 0  # Start phase while finding red cross cursor
    TARGET = 1 # Target finding phsae

# Accumulator Based Arbitration Model
# https://link.springer.com/chapter/10.1007/978-3-319-70087-8_63
# we proposed arbitration model based on prefrontal cortex
class Abam:
    def __init__(self, size, decay_rate=0.9):
        self.values = np.zeros((size), dtype=np.float32)
        self.decay_rate = decay_rate

    # accumulate max value
    # others are discounted
    def accumulate(self, values):
        max_index = np.argmax(values)
        self.values *= self.decay_rate
        self.values[max_index] += 1.0

    def output(self):
        return np.max(self.values)

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

        self.vae_cursor_accumulator = Abam(len(MODEL_PATHS.keys()))
        # replace cursor_find_accmulator with vae error accumulator
        # if vae is not consistently reliable, the scene is finding cursor
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

        # Image from Visual cortex module.
        retina_image, pixel_errors, top_errors = inputs['from_vc']

        bg_message = 0
        bg_findcursor = 0

        # accumulate reconstruction error
        errors = OrderedDict()
        names = list(MODEL_PATHS.keys())
        for name in names:
            pixel_error = pixel_errors[name]
            # scalar mean value
            errors[name] = pixel_error.mean(-1).mean(-1).mean(-1)
        # sort keys because dictionary is not consistently ordered
        sorted_errors = OrderedDict()
        for k, v in sorted(errors.items(), key=lambda x: x[0]):
            sorted_errors[k] = v
        # negate errors to accumulate most reliable one
        self.vae_cursor_accumulator.accumulate(
            -np.array(list(sorted_errors.values())))
        # current task according to most reliable vae
        current_task = names[np.argmin(list(errors.values()))]
        self.last_current_task = current_task

        # 8.5 is hyperparameter of threshold
        if self.phase == Phase.INIT:
            if self.vae_cursor_accumulator.output() < 8.5:
                self.phase = Phase.START
                bg_message = 1
                bg_findcursor = 1
        elif self.phase == Phase.START:
            if self.vae_cursor_accumulator.output() > 8.5:
                self.phase = Phase.TARGET
                bg_message = 1
            else:
                bg_findcursor = 1
        else:
            if self.vae_cursor_accumulator.output() < 8.5:
                self.phase = Phase.START

        if self.phase == Phase.INIT or self.phase == Phase.START:
            fef_message = 0
        else:
            fef_message = 1

        return dict(to_fef=(fef_message, current_task),
                    to_bg=(bg_message, current_task))
