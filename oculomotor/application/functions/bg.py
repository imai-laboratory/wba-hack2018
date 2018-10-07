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

from .constants import MODEL_PATHS

"""
This is an example implemention of BG (Basal ganglia) module.
You can change this as you like.
"""

PATH = 'models'

class BG(object):
    def __init__(self, model_name=None, skip=False):
        self.timing = brica.Timing(5, 1, 0)
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

        # build PPO agent
        # from https://github.com/takuseno/ppo
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

        config = tf.ConfigProto()
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
        pfc_data_findcursor, _, current_task = inputs['from_pfc']
        hp_data = inputs['from_hp']

        # from_hp.shape(7, 6, 8)
        hp_data_latents_selected = [buf[current_task] for buf in hp_data]

        reward = inputs['from_environment'][0]
        done = inputs['from_environment'][1]

        with self.sess.as_default():
            saliency_maps = np.array(fef_data)
            old_saliency = saliency_maps[0].reshape((1, 8, 8, 1))

            # skip cursor saliency (no need to feed into ppo)
            error_saliency = saliency_maps[2].reshape((1, 8, 8, 1))

            # (2, 8, 8, 1)
            ppo_saliency_data = np.vstack([old_saliency, error_saliency])

            # fef_latent_data.shape(1, 8) 
            # hp_data_latents_selected.shape (7, 8)
            # ppo_latent_data.shape (1, 8, 8, 1)
            ppo_latent_data = np.vstack(
                (
                    np.array(fef_latent_data)[np.newaxis, :],
                    np.array(hp_data_latents_selected)
                )
            ).reshape((1, 8, 8, 1))

            # (3, 8, 8, 1)
            ppo_input = np.vstack((ppo_saliency_data, ppo_latent_data))
            # ppo_input.shape(3, 8, 8, 1) -> (1, 8, 8, 3)
            ppo_input = np.transpose(ppo_input, [3, 1, 2, 0])

            action = self.agent.act(ppo_input, [reward], [done])[0]
            # normalize thresholds between 0.0 and 1.0
            likelihood_thresholds = (action + 1.0) / 2.0
            likelihood_thresholds = np.clip(
                likelihood_thresholds, 0.0, 1.0)
            self.step += 1
            self.last_bg_data = likelihood_thresholds

        return dict(to_pfc=None,
                    to_fef=None,
                    to_sc=likelihood_thresholds)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(PATH, datetime.datetime.now().strftime('%m%d-%s')+'.ckpt'))
