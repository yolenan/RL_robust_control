#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random

import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
                          Permute)
from keras.models import Model, Sequential
from keras.optimizers import Adam
import gym

# import deeprl_hw2 as tfrl
from dqn import DQNAgent
# from deeprl_hw2.objectives import mean_huber_loss

# from gym.envs.atari import atari_env
import core
from keras.datasets import mnist
from keras.layers import Dense, LSTM
from keras.utils import to_categorical
from keras.models import Sequential

# parameters for LSTM
nb_lstm_outputs = 30  # 神经元个数
nb_time_steps = 28  # 时间序列长度
nb_input_vector = 1  # 输入序列
acc_bound = 10


def create_lstm_model(nb_time_steps, nb_input_vector, num_actions):
    model = Sequential()
    model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
    model.add(Dense(num_actions, activation='tanh'))
    return model


# def create_model(window=4, input_shape=(84, 84), num_actions=6):
#     """Create the Q-network model.
#
#     Use Keras to construct a keras.models.Model instance (you can also
#     use the SequentialModel class).
#
#     We highly recommend that you use tf.name_scope as discussed in
#     class when creating the model and the layers. This will make it
#     far easier to understnad your network architecture if you are
#     logging with tensorboard.
#
#     Parameters
#     ----------
#     window: int
#       Each input to the network is a sequence of frames. This value
#       defines how many frames are in the sequence.
#     input_shape: tuple(int, int)
#       The expected input image size.
#     num_actions: int
#       Number of possible actions. Defined by the gym environment.
#     model_name: str
#       Useful when debugging. Makes the model show up nicer in tensorboard.
#
#     Returns
#     -------
#     keras.models.Model
#       The Q-model.
#     """
#
#     model = Sequential()
#
#     # Question 2 单一线性层
#     model.add(Flatten(input_shape=(84, 84, 4)))
#     model.add(Dense(num_actions))
#     # 卷积网络
#     # x = Convolution2D(filters=16, kernel_size=8, strides=4, activation='relu')(inputs)
#     # x = Convolution2D(filters=32, kernel_size=4, strides=2, activation='relu')(x)
#     # x = Flatten()(x)
#     # x = Dense(256, activation='relu')(x)
#     # predictions = Dense(num_actions, activation='softmax')(x)
#
#     return model


def get_output_folder(parent_dir, env_name):
    """Return save folder.

    Assumes folders in the parent_dir have suffix -run{run
    number}. Finds the highest run number and sets the output folder
    to that number + 1. This is just convenient so that if you run the
    same script multiple times tensorboard can plot all of the results
    on the same plots with different names.

    Parameters
    ----------
    parent_dir: str
      Path of the directory containing all experiment runs.

    Returns
    -------
    parent_dir/run_dir
      Path to this run's save directory.
    """
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir


def main():
    parser = argparse.ArgumentParser(description='Run DQN on Atari SpaceInvaders')
    parser.add_argument('--env', default='SpaceInvaders-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='SpaceInvaders-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')
    # parser.add_argument('--input_shape', default=(84, 84, 4), type=tuple, help='Size of each frame')

    args = parser.parse_args()

    args.output = get_output_folder(args.output, args.env)

    q_network = create_lstm_model(nb_time_steps, nb_input_vector, num_actions=gym.make(args.env).action_space.n)
    agent = DQNAgent(q_network=q_network,
                     preprocessor=core.Preprocessor(),
                     memory=core.ReplayMemory(),
                     policy=1,
                     gamma=0.1,
                     target_update_freq=100,
                     num_burn_in=100,
                     train_freq=20,
                     batch_size=32)
    agent.compile('Adam', 'mse')
    env = gym.make(args.env)
    for i_episode in range(20):
        agent.fit(env, 10 ** 6)
    env.close()
    model_json = q_network.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    main()
