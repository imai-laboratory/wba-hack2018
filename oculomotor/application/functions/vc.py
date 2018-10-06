import brica
import scipy as sp
import numpy as np
import tensorflow as tf
from .vae.train import build
from .vae import constants
# from tensorflow import keras as K


class VC(object):
    """ Visual Cortex module.
    For feature extraction we used vgg16 model in keras.
    ARGS:
        skip: skip the feature extraction if enabled
    """
    def __init__(self, skip=False):
        self.timing = brica.Timing(2, 1, 0)
        if skip:
            self.model = None
        else:
            print('loading beta VAE')
            # input_shape: (224, 224, 3)
            # self.model = K.applications.vgg16.VGG16()
            # self.model = K.applications.mobilenet.MobileNet()
            self.reconstruct, self.generate, _ = build(constants)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            self.sess.run(tf.global_variables_initializer())
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
            processed_images = (retina_image, None)
        else:
            # resizing image
            reshape_size = (64, 64, 3)
            reshaped_image = sp.misc.imresize(
                retina_image, reshape_size, interp='bilinear'
            )

            # VAE reconstruction
            with self.sess.as_default():
                reconstructed_image = self.reconstruct([reshaped_image])
            processed_images = (retina_image, reconstructed_image)


        # Current implementation just passes through input retina image to FEF and PFC.
        # TODO: change pfc fef
        return dict(to_fef=processed_images,
                    to_pfc=processed_images)
