import brica
import cv2
import scipy as sp
import numpy as np
import tensorflow as tf
from .vae.train import build
from .vae import constants
from collections import OrderedDict
# from tensorflow import keras as K

def softmax(values):
    e_x = np.exp(values - np.max(values))
    return e_x / e_x.sum(axis=0)

model_paths = OrderedDict({
    'PointToTarget': 'vae_models/pointtotarget/model.ckpt',
    'ChangeDetection': 'vae_models/changedetection/model.ckpt',
    'OddOneOut': 'vae_models/oddoneout/model.ckpt',
    'VisualSearch': 'vae_models/visualsearch/model.ckpt',
    'RandomDot': 'vae_models/randomdot/model.ckpt',
    'MultipleObject': 'vae_models/multipleobjecttracking/model.ckpt'
})

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
            tensors = OrderedDict()
            for i, (name, path) in enumerate(model_paths.items()):
                tensor = build(constants, name)
                tensors[name] = tensor
            self.reconst, self.generate = self.build_tf_call(tensors)

            config = tf.ConfigProto(
                device_count={'GPU': 1}  # NO GPU
            )
            config.gpu_options.allow_growth = True

            self.sess = tf.Session(config=config)

            # create savers
            savers = {}
            for name in model_paths.keys():
                variables = tf.get_collection(
                    tf.GraphKeys.TRAINABLE_VARIABLES, name)
                var_list = {}
                for var in variables:
                    key = var.name
                    var_list[key.replace(name, 'vae')[:-2]] = var
                savers[name] = tf.train.Saver(var_list)

            self.sess.__enter__()
            self.sess.run(tf.global_variables_initializer())

            # load all saved models
            for name, saver in savers.items():
                saver.restore(self.sess, model_paths[name])
        self.skip = skip
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
        # TODO(->smatsumori): add pretrained models
        # todo(->smatsumori): check inputs shape


        # feature extraction
        if self.skip:
            processed_images = (retina_image, None)
        else:
            # resizing image
            reshape_size = (64, 64, 3)
            reshaped_image = sp.misc.imresize(
                retina_image, reshape_size, interp='bilinear')

            # VAE reconstruction
            input_image = np.array(reshaped_image, dtype=np.float32) / 255.0
            with self.sess.as_default():
                recont_images = self.reconst([input_image])
                images = OrderedDict()
                pixel_errors = OrderedDict()
                top_errors = OrderedDict()
                for image, name in zip(recont_images, model_paths.keys()):
                    images[name] = image[0]
                    pixel_error = (image[0] - input_image) ** 2
                    pixel_errors[name] = pixel_error
                    # top 10% errors
                    flatten = np.reshape(pixel_error, [-1])
                    size = int(flatten.shape[0] * 0.01)
                    max_indices = np.argpartition(-flatten, size)[:size]
                    top_error = np.zeros(flatten.shape, dtype=np.float32)
                    top_error[max_indices] = flatten[max_indices]
                    top_error = np.reshape(top_error, pixel_error.shape)
                    # flatten channel
                    channel_mean_top_error = top_error.mean(-1)
                    flatten_mean_error = np.reshape(channel_mean_top_error, [-1])
                    flatten_mean_error[flatten_mean_error > 0] = softmax(flatten_mean_error[flatten_mean_error > 0])
                    flatten_mean_error *= 1.0 / (np.max(flatten_mean_error) + 1e-5)
                    flatten_mean_error[np.isnan(flatten_mean_error)] = 0.0
                    top_error = np.reshape(flatten_mean_error, pixel_error.shape[:-1])
                    top_error = np.array(top_error * 255.0, dtype=np.uint8)
                    top_error = cv2.resize(top_error, (128, 128))
                    top_errors[name] = np.array(top_error, dtype=np.float32) / 255.0

            processed_images = (retina_image, pixel_errors, top_errors)
            self.last_vae_reconstruction = images
            self.last_vae_top_errors = top_errors

        # Current implementation just passes through input retina image to FEF and PFC.
        # TODO: change pfc fef
        return dict(to_fef=processed_images,
                    to_pfc=processed_images)

    def build_tf_call(self, tensors):
        def reconstruct(inputs):
            feed_dict = {}
            ops = []
            for tensor in tensors.values():
                feed_dict[tensor['input']] = inputs
                feed_dict[tensor['keep_prob']] = 1.0
                feed_dict[tensor['deterministic']] = 1.0
                ops.append(tensor['reconst'])
            sess = tf.get_default_session()
            return sess.run(ops, feed_dict)

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
