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
GRID_OPTICAL_WIDTH = 64 // 4

SALIENCY_COEFF = 0.3
CURSOR_MATCH_COEFF = 1.0


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
        region_image_red = region_image_red / 255
        match = np.mean(region_image_red)
        
        # Find the maximum match value
        self.accumulate(match * CURSOR_MATCH_COEFF)
        self.expose()


class BackgroundAccumulator(ActionAccumulator):
    def __init__(self, pixel_x, pixel_y, ex, ey):
        super(BackgroundAccumulator, self).__init__(ex, ey)
        # Pixel x,y pos at left top corner of the region.
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def process(self, retina_image):
        # Crop region image (to the region)
        region_image = retina_image[self.pixel_y:self.pixel_y+GRID_WIDTH,
                                    self.pixel_x:self.pixel_x+GRID_WIDTH, :]

        sky_min = np.array([152, 189, 211], np.uint8)
        sky_max = np.array([192, 229, 255], np.uint8)
        region_image_sky = cv2.inRange(region_image, sky_min, sky_max)
        region_image_sky = region_image_sky / 255
        region_image_sky = np.ones(region_image_sky.shape) - region_image_sky
        match = np.mean(region_image_sky)

        # Find the maximum match value
        self.accumulate(match * 0.3)
        self.expose()

        
class FEF(object):
    def __init__(self):
        self.timing = brica.Timing(4, 1, 0)
        self.saliency_accumulators = []
        self.error_accumulators = []
        self.cursor_accumulators = []
        self.background_accumulators = []
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

                # cursor accumulater
                cursor_accumulator = CursorAccumulator(pixel_x, pixel_y, ex, ey, cursor_template)
                self.cursor_accumulators.append(cursor_accumulator)

                # vae error accumulator
                error_accumulator = SaliencyAccumulator(pixel_x, pixel_y, ex, ey)
                self.error_accumulators.append(error_accumulator)

                # background accumulator
                background_accumulator = BackgroundAccumulator(pixel_x, pixel_y, ex, ey)
                self.background_accumulators.append(background_accumulator)

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

        # task (string)
        phase, task = inputs['from_pfc']
        saliency_map, optical_flow = inputs['from_lip']
        # latents.values().shape: (6, 8)
        retina_image, pixel_errors, top_errors, dc_latents = inputs['from_vc']

        opticalxflow = optical_flow[32:97,32:97, 0]
        opticalyflow = optical_flow[32:97,32:97, 1]

        rx = opticalxflow.reshape(-1)
        ry = opticalyflow.reshape(-1)        

        angle = 0
        for i, (x, y) in enumerate(zip(rx[1:], ry[1:])):
            if i == 1:
                angle = calc_angle(x, y)
            else:
                angle_ = calc_angle(x, y)
                angle = np.max([angle, angle_]) - (np.abs(angle - angle_) / 2)

        rad = np.array(range(0, 360, 45))
        ind = np.argmax(-np.abs(rad - angle))

        a0 = np.zeros([8, 8])
        a0[3:5, 6:8] = 1
        
        a1 = np.zeros([8, 8])
        a1[0:2, 6:8] = 1

        a2 = np.zeros([8, 8])
        a2[0:2, 3:5] = 1

        a3 = np.zeros([8, 8])
        a3[0:2, 0:2] = 1
        
        a4 = np.zeros([8, 8])
        a4[3:5, 0:2] = 1

        a5 = np.zeros([8, 8])
        a5[6:8, 0:2] = 1

        a6 = np.zeros([8, 8])
        a6[6:8, 3:5] = 1
        
        a7 = np.zeros([8, 8])
        a7[6:8, 6:8] = 1

        a = [a0, a1, a2, a3, a4, a5, a6, a7]
        arrow = a[ind]

        ax = np.zeros([8, 8])
        ay = np.zeros([8, 8])
        for ix in range(GRID_DIVISION):
            pixel_x = GRID_WIDTH * ix
            cx = 2.0 / GRID_DIVISION * (ix + 0.5) - 1.0
            for iy in range(GRID_DIVISION):
                pixel_y = GRID_WIDTH * iy
                cy = 2.0 / GRID_DIVISION * (iy + 0.5) - 1.0
                
                ax[ix, iy] = -cx
                ay[ix, iy] = -cy

        arrow_output = np.concatenate([arrow, ax], axis=1)
        arrow_output = np.concatenate([arrow_output, ay], axis=1)
                
        # TODO: 領野をまたいだ共通phaseをどう定義するか？
        # Update accumulator
        # phase == 0 finding cursor
        if phase == 0:
            for cursor_accumulator in self.cursor_accumulators:
                cursor_accumulator.process(retina_image)
        else:
            for saliency_accumulator in self.saliency_accumulators:
                # decrease basic saliency to prioritize vae saliency
                saliency_accumulator.process(saliency_map / 2.0)

            for error_accumulator in self.error_accumulators:
                # accumulate current task error
                error_accumulator.process(top_errors[task] * 5.0)
                
            for background_accumulator in self.background_accumulators:
                background_accumulator.process(retina_image)
                
        # select latents according to tasks
        # to_bg_latent.shape: (1, 8)
        to_bg_latent = dc_latents[task]
                
        # discount accumulator
        # TODO: discount after collecting outputs?
        for saliency_accumulator in self.saliency_accumulators:
            saliency_accumulator.post_process()

        for cursor_accumulator in self.cursor_accumulators:
            cursor_accumulator.post_process()
       
        for error_accumulator in self.error_accumulators:
            error_accumulator.post_process()

        for background_accumulator in self.background_accumulators:
            background_accumulator.post_process()

        # collect all outputs (Nx64, 3) -> (n, 64) -> (n, 8, 8)
        # where N is a features for bg input
        output = self._collect_output()
        reshaped_output = np.array(output).reshape(
            3, 64, 3)[:, :, 0].reshape(3, 8, 8).tolist()

        # TODO: change shape output
#        output.append(np.expand_dims(opticalxflow.reshape(-1), axis=1))
#        output.append(np.expand_dims(opticalyflow.reshape(-1), axis=1))
#        output = np.array(output, dtype=np.float32)

        return dict(to_pfc=None,
                    to_bg=(reshaped_output, to_bg_latent),
                    to_sc=output,
                    to_cb=None)

    def _collect_output(self):
        output = []
        for saliency_accumulator in self.saliency_accumulators:
            output.append(saliency_accumulator.output)

        for cursor_accumulator in self.cursor_accumulators:
            output.append(cursor_accumulator.output)

 
        for error_accumulator in self.error_accumulators:
            output.append(error_accumulator.output)

        output = np.array(output)
        background_output = []
        for background_accumulator in self.background_accumulators:
            background_output.append(background_accumulator.output[0])
        background_output = np.array(background_output)
        if np.mean(background_output) > 0.5:
            background_output = np.zeros(background_output.shape)
        output[64:128][:,0] += background_output
        #output[64:128][:,0] = np.abs(output[64:128][:,0])
        output[64:128][:,0] = np.clip(output[64:128][:,0], 0, 1)

        return output


def calc_angle(x, y):
    u = np.array([1, 0])
    v = np.array([x, y])
    i = np.inner(u, v)
    n = np.linalg.norm(u) * np.linalg.norm(v)

    c = i / (n + 1e-19)

    return np.rad2deg(np.arccos(np.clip(c, -1.0, 1.0)))
