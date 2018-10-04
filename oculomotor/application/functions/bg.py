import numpy as np
import tensorflow as tf
import brica
import ppo.constants as constants

from ppo.agent import Agent
from ppo.network import make_network
from ppo.scheduler import LinearScheduler, ConstantScheduler

"""
This is an example implemention of BG (Basal ganglia) module.
You can change this as you like.
"""

class BG(object):
    def __init__(self):
        self.timing = brica.Timing(5, 1, 0)
        state_shape = [constants.STATE_SHAPE]
        num_actions = 1

        # create network function
        model = make_network(
            constants.CONVS, constants.FCS, use_lstm=constants.LSTM,
            padding=constants.PADDING, continuous=True)

        # scheduled paramters
        if constants.LR_DECAY == 'linear':
            lr = LinearScheduler(constants.LR, constants.FINAL_STEP, 'lr')
            epsilon = LinearScheduler(
                constants.EPSILON, constants.FINAL_STEP, 'epsilon')
        else:
            lr = ConstantScheduler(constants.LR, 'lr')
            epsilon = ConstantScheduler(constants.EPSILON, 'epsilon')

        self.agent = Agent(
            model,
            num_actions,
            nenvs=1,
            lr=lr,
            epsilon=epsilon,
            gamma=constants.GAMMA,
            lam=constants.LAM,
            lstm_unit=constants.LSTM_UNIT,
            value_factor=constants.VALUE_FACTOR,
            entropy_factor=constants.ENTROPY_FACTOR,
            time_horizon=constants.TIME_HORIZON,
            batch_size=constants.BATCH_SIZE,
            grad_clip=constants.GRAD_CLIP,
            state_shape=state_shape,
            epoch=constants.EPOCH,
            use_lstm=constants.LSTM,
            continuous=True,
            upper_bound=1.0
        )

        self.sess = tf.Session()
        self.sess.__enter__()
        self.sess.run(tf.global_variables_initializer())

    def __call__(self, inputs):
        self.sess.__enter__()

        if 'from_environment' not in inputs:
            raise Exception('BG did not recieve from Environment')
        if 'from_pfc' not in inputs:
            raise Exception('BG did not recieve from PFC')
        if 'from_fef' not in inputs:
            raise Exception('BG did not recieve from FEF')

        fef_data = inputs['from_fef']

        accmulator_size = len(fef_data)

        # Set threshold as 0.1 (as dummy test)
        likelihood_thresholds = np.ones([accmulator_size], dtype=np.float32) * 0.3

        # self.agent.act(input, reward, done)

        return dict(to_pfc=None,
                    to_fef=None,
                    to_sc=likelihood_thresholds)
