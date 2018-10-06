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
    def __init__(self, model_name, skip=False):
        self.timing = brica.Timing(5, 1, 0)
        self.skip = skip
        self.step = 0
        self.model_name = None
        if not skip:
            self.__initialize_rl()
        self.last_bg_data = None

    def __initialize_rl(self):
        # TODO: do we need convs?
        # state_shape = [ppconsts.STATE_SHAPE]  # state_shape = input shape of the network
        # state_shape = (128, 3)
        state_shape = [128, 3]
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
            state_shape=state_shape,
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
                model_name += '.ckpt'
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

        fef_data = inputs['from_fef']
        pfc_data = inputs['from_pfc']
        if 0 < pfc_data:
            print("\033 internal reward!! \033[0m")
        reward, done = inputs['from_environment'][0] + pfc_data, inputs['from_environment'][1]

        # default FEF shape.(128, 3) -> (64, 3)
        # psudo action space (can we pass images or features?)
        if self.skip:
            # action space will be fixed
            accmulator_size = len(fef_data)
            # Set threshold as 0.1 (as dummy test)
            likelihood_thresholds = np.ones([accmulator_size], dtype=np.float32) * 0.3
        else:
            with self.sess.as_default():
                # TODO(->smatsumori): check input shape
                fef_data = np.array(fef_data)[np.newaxis, :, :]
                likelihood_thresholds = self.agent.act(fef_data, [reward], [done])[0]
                self.step += 1
                self.last_bg_data = likelihood_thresholds

        return dict(to_pfc=None,
                    to_fef=None,
                    to_sc=likelihood_thresholds)

    def save_model(self):
        self.saver.save(self.sess, os.path.join(PATH, datetime.datetime.now().strftime('%m%d-%s')+'.ckpt'))
