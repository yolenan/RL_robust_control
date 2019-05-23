#!/usr/bin/env python
"""Run Atari Environment with DQN."""
# import argparse
# import os
# import random
#
# import numpy as np
# import tensorflow as tf
# from keras.layers import (Activation, Convolution2D, Dense, Flatten, Input,
#                           Permute)
# from keras.models import Model, Sequential
# from keras.optimizers import Adam
from DnsCarFollowENV2 import VehicleFollowingENV

# import deeprl_hw2 as tfrl
from dqn import DQNAgent
# from deeprl_hw2.objectives import mean_huber_loss

# from gym.envs.atari import atari_env
import core
from keras.datasets import mnist
from keras.layers import Dense, LSTM, Embedding
from keras.utils import to_categorical
from keras.models import Sequential

# parameters for LSTM
nb_lstm_outputs = 32  # 神经元个数
nb_time_steps = 1  # 时间序列长度
nb_input_vector = 4  # 输入序列
acc_bound = 10


def create_lstm_model(nb_time_steps, nb_input_vector, num_actions):
    model = Sequential()
    model.add(Embedding(input_dim=nb_input_vector, output_dim=32))
    # model.add(LSTM(units=nb_lstm_outputs, input_shape=(nb_time_steps, nb_input_vector)))
    model.add(LSTM(nb_lstm_outputs))
    model.add(Dense(num_actions, activation='tanh'))
    return model


def main():
    # vehicle_network
    veh_network = create_lstm_model(nb_time_steps, nb_input_vector, num_actions=4)
    # Attacker network
    # att_network = create_lstm_model(nb_time_steps, nb_input_vector, num_actions=4)
    veh_agent = DQNAgent(q_network=veh_network,
                         q_network2=veh_network,
                         preprocessor=core.Preprocessor(),
                         memory=core.ReplayMemory(),
                         policy=1,
                         gamma=0.1,
                         target_update_freq=100,
                         num_burn_in=100,
                         train_freq=20,
                         batch_size=32)
    # att_agent = DQNAgent(q_network=att_network,
    #                      q_network2=att_network,
    #                      preprocessor=core.Preprocessor(),
    #                      memory=core.ReplayMemory(),
    #                      policy=1,
    #                      gamma=0.1,
    #                      target_update_freq=100,
    #                      num_burn_in=100,
    #                      train_freq=20,
    #                      batch_size=32)
    veh_agent.compile('Adam', 'mse')
    # att_agent.compile('Adam', 'mse')
    env = VehicleFollowingENV()
    for i_episode in range(20):
        veh_agent.fit(env, 10 ** 6)
        # att_agent.fit(env, 10 ** 6)
    # env.close()
    model_json = veh_network.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)


if __name__ == '__main__':
    main()
