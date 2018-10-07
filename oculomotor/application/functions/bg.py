import os
import numpy as np
import tensorflow as tf
import brica
from .ppo import constants as ppconsts
from . import constants as consts

import datetime
from .ppo.agent import Agent
from .ppo.network import make_network
from .ppo.scheduler import LinearScheduler, ConstantScheduler

"""
This is an example implemention of BG (Basal ganglia) module.
You can change this as you like.
"""

PATH = 'models'

class BG(object):
    def __init__(self, model_name=None, skip=False):
        self.timing = brica.Timing(5, 1, 0)
        self.skip = skip
        self.step = 0
        self.model_name = model_name
        if model_name is not None:
            print('loading model: {}'.format(model_name))
        if not skip:
            self.__initialize_rl()
        self.last_bg_data = None

    def __initialize_rl(self):
        num_actions = consts.NUM_ACTIONS

        # TODO(->smatsumori): load from saved models
        # create network function
        model = make_network(
            ppconsts.CONVS, ppconsts.FCS, use_lstm=ppconsts.LSTM,
            padding=ppconsts.PADDING, continuous=True)

        # scheduled paramters
        if ppconsts.LR_DECAY == 'linear':
            lr = LinearScheduler(ppconsts.LR, ppconsts.FINAL_STEP, 'lr')
            epsilon = LinearScheduler(
                ppconsts.EPSILON, ppconsts.FINAL_STEP, 'epsilon')
        else:
            lr = ConstantScheduler(ppconsts.LR, 'lr')
            epsilon = ConstantScheduler(ppconsts.EPSILON, 'epsilon')

        self.agent = Agent(
            model,
            num_actions,
            nenvs=1,
            lr=lr,
            epsilon=epsilon,
            gamma=ppconsts.GAMMA,
            lam=ppconsts.LAM,
            lstm_unit=ppconsts.LSTM_UNIT,
            value_factor=ppconsts.VALUE_FACTOR,
            entropy_factor=ppconsts.ENTROPY_FACTOR,
            time_horizon=ppconsts.TIME_HORIZON,
            batch_size=ppconsts.BATCH_SIZE,
            grad_clip=ppconsts.GRAD_CLIP,
            state_shape=ppconsts.STATE_SHAPE,
            epoch=ppconsts.EPOCH,
            use_lstm=ppconsts.LSTM,
            continuous=True,
            upper_bound=1.0
        )

        config = tf.ConfigProto(
            device_count={'GPU': 2}  # NO GPU
        )
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.sess.__enter__()
        self.saver = tf.train.Saver()
        if self.model_name:
            if not self.model_name.endswith('.ckpt'):
                self.model_name += '.ckpt'
            self.saver.restore(self.sess, os.path.join(PATH, self.model_name))
        self.sess.run(tf.global_variables_initializer())


    def __call__(self, inputs, update=False):
        # TODO; update params
        # update True when to update parameters

        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')

        # fef_latent_data.shape: (1, 8)
        fef_data, fef_latent_data = inputs['from_fef']
        pfc_data = inputs['from_pfc'][0]
        pfc_data_findcursor, _, current_task = inputs['from_pfc']
        hp_data_latents_buffers = inputs['from_hp']  # .shape(7, 6, 8)

        # TODO(->smatsumori): selecet episodes from current tasks
        reward, done = inputs['from_environment'][0], inputs['from_environment'][1]

        # default FEF shape.(128, 3) -> (64, 3)
        # psudo action space (can we pass images or features?)
        if self.skip or pfc_data_findcursor == 1:
            # action space will be fixed
            saliency_maps = np.array(fef_data)
            accmulator_size = saliency_maps.size
            # Set threshold as 0.1 (as dummy test)
            likelihood_thresholds = np.ones(
                [accmulator_size], dtype=np.float32) * 0.3
        else:
            with self.sess.as_default():
                # TODO(->seno): change order
                # saliency_maps.shape (3, 8, 8) (saliency, cursor, error)
                saliency_maps = np.array(fef_data)
                old_saliency = saliency_maps[0].reshape((1, 8, 8, 1))

                # skip cursor saliency (no need to feed into ppo)
                error_saliency = saliency_maps[2].reshape((1, 8, 8, 1))
                ppo_input = np.vstack([old_saliency, error_saliency])

                # ppo_input.shape: (1, 8, 8, 2)
                ppo_input = np.transpose(ppo_input, [3, 1, 2, 0])
                likelihood_thresholds = (
                    self.agent.act(ppo_input, [reward], [done])[0] + 1.0) / 2.0
                likelihood_thresholds = np.clip(likelihood_thresholds, 0.0, 1.0)
                self.step += 1
                self.last_bg_data = likelihood_thresholds

        return dict(to_pfc=None,
                    to_fef=None,
                    to_sc=likelihood_thresholds)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(PATH, datetime.datetime.now().strftime('%m%d-%s')+'.ckpt'))
