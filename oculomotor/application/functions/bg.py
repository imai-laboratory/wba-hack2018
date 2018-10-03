import numpy as np
import tensorflow as tf
import brica
import ppo.constants as ppconsts
import constants as consts

from ppo.agent import Agent
from ppo.network import make_network
from ppo.scheduler import LinearScheduler, ConstantScheduler

"""
This is an example implemention of BG (Basal ganglia) module.
You can change this as you like.
"""

class BG(object):
    def __init__(self, num_actions):
        self.timing = brica.Timing(5, 1, 0)
        state_shape = [ppconsts.STATE_SHAPE]
        num_actions = consts.NUM_ACTIONS

        # TODO(->smatsumori): load from saved models
        # create network function
        model = make_network(
            ppconsts.CONVS, ppconsts.FCS, use_lstm=ppconsts.LSTM,
            padding=ppconsts.PADDING, continuous=True)

        # scheduled paramters
        lr = tf.Variable(ppconsts.LR)
        decayed_lr = tf.placeholder(tf.float32)
        decay_lr_op = lr.assign(decayed_lr)
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
        reward = inputs['from_environment']

        # psudo action space (can we pass images or features?)
        # action space will be fixed
        accmulator_size = len(fef_data)

        # Set threshold as 0.1 (as dummy test)
        likelihood_thresholds = np.ones([accmulator_size], dtype=np.float32) * 0.3

        # self.agent.act(input, reward, done)

        # self.agent.receive_next(next_input, next_reward, next_done, update=False)

        # when update parameters
        # self.agent.receive_next(next_input, next_reward, next_done, update=True)

        return dict(to_pfc=None,
                    to_fef=None,
                    to_sc=likelihood_thresholds)
