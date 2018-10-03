import brica
import scipy as sp
import numpy as np
from tensorflow import keras as K


class VC(object):
    """ Visual Cortex module.
    For feature extraction we used vgg16 model in keras.
    ARGS:
        skip: skip the feature extraction if enabled
    """
    def __init__(self, skip=True):
        self.timing = brica.Timing(2, 1, 0)
        if skip:
            self.model = None
        else:
            # input_shape: (224, 224, 3)
            # self.model = K.applications.vgg16.VGG16()
            self.model = K.applications.mobilenet.MobileNet()
        self.skip = skip

    def __call__(self, inputs):
        """
        ARGS:
            inputs: inputs from retina.
                .shape: (128, 128, 3)
        """
        if 'from_retina' not in inputs:
            raise Exception('VC did not recieve from Retina')

        retina_image = inputs['from_retina']
        # TODO(->smatsumori): add pretrained models
        # todo(->smatsumori): check inputs shape

        # feature extraction
        if self.skip:
            processed_image = retina_image
        else:
            reshaped_image = sp.misc.imresize(
                retina_image, (224, 224, 3), interp='bilinear'
            )
            print('reshaped', reshaped_image.shape)
            # feature_map = self.model.predict(reshaped_image)
            # print('feature', feature_map)
            processed_image = retina_image


        # Current implementation just passes through input retina image to FEF and PFC.
        return dict(to_fef=processed_image,
                    to_pfc=processed_image)
