import brica
import scipy as sp
import numpy as np
import tensorflow as tf
from .vae.train import build
from .vae import constants
from collections import OrderedDict
# from tensorflow import keras as K


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
        # TODO: remove skip
        if self.skip:
            processed_images = (retina_image, None)
        else:
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

                # recont_images.shape: (6, 128, 128, 3)
                # latents.shape: (6, 8)
                recont_images, latents = reconstructed[:6], reconstruct[6:]
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
                    top_error[max_indices] = 1.0
                    top_errors[name] = np.reshape(top_error, pixel_error.shape)

            to_fef = (retina_image, pixel_errors, top_errors, latents)
            to_pfc = (retina_image, pixel_errors, top_errors)
            to_hp = (latents)
            self.last_vae_reconstruction = images
            self.last_vae_top_errors = top_errors

        # Current implementation just passes through input retina image to FEF and PFC.
        # TODO: change pfc fef
        return dict(to_fef=to_fef, to_pfc=to_pfc, to_hp=to_hp)

    def build_tf_call(self, tensors):
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
