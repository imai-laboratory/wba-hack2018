import brica
import cv2
import scipy as sp
import numpy as np
import tensorflow as tf
from .vae.train import build
from .vae import constants
from .constants import MODEL_PATHS
from .utils import softmax
from collections import OrderedDict
# from tensorflow import keras as K


class VC(object):
    """ Visual Cortex module.
    For feature extraction we used vgg16 model in keras.
    ARGS:
        skip: skip the feature extraction if enabled
    """
    def __init__(self, skip=False):
        self.timing = brica.Timing(2, 1, 0)

        # build vae graphs on each task
        tensors = OrderedDict()
        for i, (name, path) in enumerate(MODEL_PATHS.items()):
            tensor = build(constants, name)
            tensors[name] = tensor
        # aggregate all tensors in single inference
        self.reconst, self.generate = self.build_tf_call(tensors)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)

        # create savers on each graph
        savers = {}
        for name in MODEL_PATHS.keys():
            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES, name)
            var_list = {}
            for var in variables:
                key = var.name
                # replace vae in namespace with task name
                var_list[key.replace(name, 'vae')[:-2]] = var
            savers[name] = tf.train.Saver(var_list)

        self.sess.__enter__()
        self.sess.run(tf.global_variables_initializer())

        print('loading beta VAE')
        # load all saved models
        for name, saver in savers.items():
            saver.restore(self.sess, MODEL_PATHS[name])

        self.last_vae_reconstruction = None
        self.last_vae_top_errors = None

    def __call__(self, inputs):
        """
        ARGS:
            inputs: inputs from retina.
                .shape: (128, 128, 3)
        """
        if 'from_retina' not in inputs:
            raise Exception('VC did not recieve from Retina')

        retina_image = inputs['from_retina']

        # feature extraction
        # resizing image
        reshape_size = (64, 64, 3)
        reshaped_image = sp.misc.imresize(
            retina_image, reshape_size, interp='bilinear')

        # VAE reconstruction and get latent parameters
        input_image = np.array(reshaped_image, dtype=np.float32) / 255.0
        with self.sess.as_default():
            # WARN: using magi no
            # reconstructed.len: 12
            reconstructed = self.reconst([input_image])

            # recont_images.shape: (6, 64, 64, 3)
            # latents.shape: (6, 8)
            recont_images, latents = reconstructed[:6], reconstructed[6:]
            images = OrderedDict()
            pixel_errors = OrderedDict()
            top_errors = OrderedDict()
            dc_latents = {}
            for image, latent, name\
                    in zip(recont_images, latents, MODEL_PATHS.keys()):
                images[name] = image[0]
                pixel_error = (image[0] - input_image) ** 2
                pixel_errors[name] = pixel_error
                # top 10% errors
                flatten = np.reshape(pixel_error, [-1])
                size = int(flatten.shape[0] * 0.01)
                max_indices = np.argpartition(-flatten, size)[:size]
                top_error = np.zeros(flatten.shape, dtype=np.float32)
                top_error[max_indices] = flatten[max_indices]
                # mean RGB channel
                top_error = np.reshape(top_error, pixel_error.shape)
                channel_mean_top_error = top_error.mean(-1)
                # flatten errors to normalize
                flatten_mean_error = np.reshape(channel_mean_top_error, [-1])
                flatten_mean_error[flatten_mean_error > 0] = softmax(
                    flatten_mean_error[flatten_mean_error > 0])
                flatten_mean_error *= 1.0 / (np.max(flatten_mean_error) + 1e-5)
                flatten_mean_error[np.isnan(flatten_mean_error)] = 0.0
                # reshape to (64, 64)
                top_error = np.reshape(flatten_mean_error, pixel_error.shape[:-1])
                # multiply 255 to correctly resize error as an image
                top_error = np.array(top_error * 255.0, dtype=np.uint8)
                top_error = cv2.resize(top_error, (128, 128))
                top_errors[name] = np.array(top_error, dtype=np.float32) / 255.0

                dc_latents[name] = latent[0]

            to_fef = (retina_image, pixel_errors, top_errors, dc_latents)
            to_pfc = (retina_image, pixel_errors, top_errors)
            to_hp = (dc_latents)
            self.last_vae_reconstruction = images
            self.last_vae_top_errors = top_errors

        # Current implementation just passes through input retina image to FEF and PFC.
        # TODO: change pfc fef
        return dict(to_fef=to_fef, to_pfc=to_pfc, to_hp=to_hp)

    # aggregate function to infere in a single sesion.run
    def build_tf_call(self, tensors):
        # reconstruct input image
        def reconstruct(inputs):
            feed_dict = {}
            ops = []

            # tensor.values for each tasks
            for tensor in tensors.values():
                feed_dict[tensor['input']] = inputs
                feed_dict[tensor['keep_prob']] = 1.0
                feed_dict[tensor['deterministic']] = 1.0
                ops.append(tensor['reconst'])

            # get latent parameters
            # mu.output.shape (6, 8)
            for tensor in tensors.values():
                ops.append(tensor['mu'])

            # ops.len == 12
            sess = tf.get_default_session()
            return sess.run(ops, feed_dict)

        # generate image from latent variable
        def generate(latent):
            feed_dict = {}
            ops = []
            for tensor in tensors.values():
                feed_dict[tensor['latent']] = latent
                feed_dict[tensor['keep_prob']] = 1.0
                feed_dict[tensor['deterministic']] = 1.0
                ops.append(tensor['generate'])
            sess = tf.get_default_session()
            return sess.run(ops, feed_dict)

        return reconstruct, generate
