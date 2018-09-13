import threading
import multiprocessing
import argparse
import cv2
import gym
import copy
import os
import time
import atari_constants
import numpy as np
import tensorflow as tf

from oculoenv import PointToTargetContent, Environment
from rlsaber.log import TfBoardLogger, dump_constants
from rlsaber.trainer import BatchTrainer
from rlsaber.env import EnvWrapper, BatchEnvWrapper, NoopResetEnv, EpisodicLifeEnv, MaxAndSkipEnv
from network import make_network
from agent import Agent
from datetime import datetime


def main():
    date = datetime.now().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--load', type=str)
    parser.add_argument('--logdir', type=str, default=date)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    outdir = os.path.join(os.path.dirname(__file__), 'results/' + args.logdir)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    logdir = os.path.join(os.path.dirname(__file__), 'logs/' + args.logdir)

    content = PointToTargetContent()
    tmp_env = Environment(content)

    constants = atari_constants
    num_actions = 2
    state_shape = constants.STATE_SHAPE + [3]
    def state_preprocess(state):
        state = state['screen']
        state = cv2.resize(state, tuple(constants.STATE_SHAPE))
        state = np.array(state, dtype=np.float32)
        return state / 255.0
    # (window_size, H, W) -> (H, W, window_size)
    phi = lambda s: s[0]

    # save settings
    dump_constants(constants, os.path.join(outdir, 'constants.json'))

    sess = tf.Session()
    sess.__enter__()

    model = make_network(
        constants.CONVS, constants.FCS,
        lstm=constants.LSTM, padding=constants.PADDING)

    lr = tf.Variable(constants.LR)
    decayed_lr = tf.placeholder(tf.float32)
    decay_lr_op = lr.assign(decayed_lr)
    optimizer = tf.train.RMSPropOptimizer(lr, decay=0.99, epsilon=1e-5)

    agent = Agent(
        model,
        num_actions,
        optimizer,
        nenvs=constants.ACTORS,
        gamma=constants.GAMMA,
        lstm_unit=constants.LSTM_UNIT,
        time_horizon=constants.TIME_HORIZON,
        value_factor=constants.VALUE_FACTOR,
        entropy_factor=constants.ENTROPY_FACTOR,
        grad_clip=constants.GRAD_CLIP,
        state_shape=state_shape,
        phi=phi
    )

    saver = tf.train.Saver()
    if args.load:
        saver.restore(sess, args.load)

    # create environemtns
    envs = []
    for i in range(constants.ACTORS):
        content = PointToTargetContent()
        env = Environment(content)
        env.observation_space = np.zeros(state_shape, dtype=np.float32)
        env.action_space = None
        wrapped_env = EnvWrapper(
            env,
            r_preprocess=lambda r: np.clip(r, -1.0, 1.0),
            s_preprocess=state_preprocess
        ) 
        envs.append(wrapped_env)
    batch_env = BatchEnvWrapper(envs)

    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter(logdir, sess.graph)
    logger = TfBoardLogger(summary_writer)
    logger.register('reward', dtype=tf.float32)
    end_episode = lambda r, s, e: logger.plot('reward', r, s)

    def after_action(state, reward, global_step, local_step):
        if constants.LR_DECAY == 'linear':
            decay = 1.0 - (float(global_step) / constants.FINAL_STEP)
            if decay < 0.0:
                decay = 0.0
            sess.run(decay_lr_op, feed_dict={decayed_lr: constants.LR * decay})
        if global_step % 10 ** 6 == 0:
            path = os.path.join(outdir, 'model.ckpt')
            saver.save(sess, path, global_step=global_step)

    trainer = BatchTrainer(
        env=batch_env,
        agent=agent,
        render=args.render,
        state_shape=state_shape[:-1],
        state_window=constants.STATE_WINDOW,
        time_horizon=constants.TIME_HORIZON,
        final_step=constants.FINAL_STEP,
        after_action=after_action,
        end_episode=end_episode,
        training=not args.demo
    )
    trainer.start()

if __name__ == '__main__':
    main()