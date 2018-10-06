# -*- coding: utf-8 -*-
import os
import cv2
import math
import numpy as np

import brica
from .utils import load_image

"""
This is an example implemention of FEF (Frontal Eye Field) module.
You can change this as you like.
"""

GRID_DIVISION = 8
GRID_WIDTH = 128 // GRID_DIVISION

CURSOR_MATCH_COEFF = 0.3
SALIENCY_COEFF = 0.3


class ActionAccumulator(object):
    """
    Sample implementation of an accmulator.
    """
    def __init__(self, ex, ey, decay_rate=0.9):
        """
        Arguments:
          ex: Float eye move dir x
          ey: Float eye move dir Y
        """
        # Accumulated likehilood
        self.likelihood = 0.0
        # Eye movment
        self.ex = ex
        self.ey = ey
        # Decay rate of likehilood
        self.decay_rate = decay_rate

        # Connected accmulators
        self.target_accmulators = []

    def accumulate(self, value):
        self.likelihood += value

    def expose(self):
        # Sample implementation of accmulator connection.
        # Send accmulated likelihood to another accmulator.
        for target_accmulator in self.target_accmulators:
            weight = 0.01
            target_accmulator.accumulate(self.likelihood * weight)

    def post_process(self):
        # Clip likelihood
        self.likelihood = np.clip(self.likelihood, 0.0, 1.0)
        # Decay likelihood
        self.likelihood *= self.decay_rate

    def reset(self):
        self.likelihood = 0.0

    def connect_to(self, target_accmulator):
        self.target_accmulators.append(target_accmulator)

    @property
    def output(self):
        return [self.likelihood, self.ex, self.ey]


class SaliencyAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey):
        super(SaliencyAccumulator, self).__init__(ex, ey, decay_rate=0.85)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def process(self, saliency_map):
        # Crop region image
        region_saliency = saliency_map[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                       self.pixel_x:self.pixel_x+GRID_WIDTH]
        average_saliency = np.mean(region_saliency)
        self.accumulate(average_saliency * SALIENCY_COEFF)
        self.expose()


class CursorAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey, cursor_template):
        super(CursorAccumulator, self).__init__(ex, ey)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.cursor_template = cursor_template

    def process(self, retina_image):
        # Crop region image (to the region)
        region_image = retina_image[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                    self.pixel_x:self.pixel_x+GRID_WIDTH, :]
        
        red_min = np.array([150, 0, 0], np.uint8)
        red_max = np.array([255, 100, 100], np.uint8)
        region_image_red = cv2.inRange(region_image, red_min, red_max)
        match = np.sum(region_image_red)

        '''        
        region_image = np.array(region_image)
        region_image = region_image.reshape((-1, 3))
        region_image = np.mean(region_image, axis=0)
        match = np.inner(region_image, np.array([0, 255, 0])) // 65000

        # add
        red_min = np.array([0, 0, 0], np.uint8)
        red_max = np.array([255, 100, 100], np.uint8)

        red_min = np.array([200, 0, 0], np.uint8)
        red_max = np.array([255, 0, 0], np.uint8)

        region_image_red = cv2.inRange(region_image, red_min, red_max)
        region_image_red = cv2.cvtColor(region_image_red, cv2.COLOR_GRAY2RGB)
        cursor_template_red = cv2.inRange(self.cursor_template, red_min, red_max)
        cursor_template_red = cv2.cvtColor(cursor_template_red, cv2.COLOR_GRAY2RGB)

        # Calculate template matching
        #match = cv2.matchTemplate(region_image, self.cursor_template, cv2.TM_CCOEFF_NORMED)
        match = cv2.matchTemplate(region_image_red, cursor_template_red, cv2.TM_CCOEFF_NORMED)
        '''

        # Find the maximum match value
        match_rate = np.max(match)
        self.accumulate(match_rate * CURSOR_MATCH_COEFF)
        self.expose()


class FEF(object):
    def __init__(self):
        self.timing = brica.Timing(4, 1, 0)
        self.saliency_accumulators = []
        self.cursor_accumulators = []
        cursor_template = load_image("data/debug_cursor_template_w.png")

        # devide and create accumulators for each region
        for ix in range(GRID_DIVISION):
            pixel_x = GRID_WIDTH * ix
            cx = 2.0 / GRID_DIVISION * (ix + 0.5) - 1.0
            for iy in range(GRID_DIVISION):
                pixel_y = GRID_WIDTH * iy
                cy = 2.0 / GRID_DIVISION * (iy + 0.5) - 1.0

                ex = -cx
                ey = -cy

                # accumulators shape (GRID_DIVISION**2, ) * 2
                saliency_accumulator = SaliencyAccumulator(pixel_x, pixel_y, ex, ey)
                self.saliency_accumulators.append(saliency_accumulator)

                # TODO: want to remove this!
                cursor_accumulator = CursorAccumulator(pixel_x, pixel_y, ex, ey,
                                                       cursor_template)
                self.cursor_accumulators.append(cursor_accumulator)

        # TODO: check what means accumulator connection
        # Accmulator connection sample
        # self.saliency_accumulators[0].connect_to(self.saliency_accumulators[1])

    def __call__(self, inputs):
        if 'from_lip' not in inputs:
            raise Exception('FEF did not recieve from LIP')
        if 'from_vc' not in inputs:
            raise Exception('FEF did not recieve from VC')
        if 'from_pfc' not in inputs:
            raise Exception('FEF did not recieve from PFC')
        if 'from_bg' not in inputs:
            raise Exception('FEF did not recieve from BG')

        phase = inputs['from_pfc']
        saliency_map, optical_flow = inputs['from_lip']
        retina_image, reconstructed_image = inputs['from_vc']

        # TODO: 領野をまたいだ共通phaseをどう定義するか？
        # Update accumulator
        # phase == 0 finding cursor
        if phase == 0:
            for cursor_accumulator in self.cursor_accumulators:
                cursor_accumulator.process(retina_image)
        else:
            for saliency_accumulator in self.saliency_accumulators:
                saliency_accumulator.process(saliency_map)

        # discount accumulator
        # TODO: discount after collecting outputs?
        for saliency_accumulator in self.saliency_accumulators:
            saliency_accumulator.post_process()
        for cursor_accumulator in self.cursor_accumulators:
            cursor_accumulator.post_process()

        # collect all outputs
        output = self._collect_output()

        # TODO: pass feature extracted to bg?
        # NOTE: do not change the output size of to_sc (=action space of BG)
        return dict(to_pfc=None,
                    to_bg=output,
                    to_sc=output,
                    to_cb=None)

    def _collect_output(self):
        output = []
        for saliency_accumulator in self.saliency_accumulators:
            output.append(saliency_accumulator.output)
        for cursor_accumulator in self.cursor_accumulators:
            output.append(cursor_accumulator.output)
        return np.array(output, dtype=np.float32)
