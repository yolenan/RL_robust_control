#!/usr/bin/env python
import torch
from naf import NAF
# from tensorboardX import SummaryWriter
from replay_memory import ReplayMemory, Transition
import numpy as np
import random
from DnsCarFollowENV2 import VehicleFollowingENV
from ounoise import OUNoise
from Supervised_Learning import create_SL_model
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
import argparse
import os
import matplotlib.pyplot as plt
import pandas as pd

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.manual_seed(1234)
np.random.seed(1234)
# is_cuda = torch.cuda.is_available()
is_cuda = False

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=True)
parser.add_argument('--noise_scale', type=float, default=0.3, metavar='G',
                    help='initial noise scale (default: 0.3)')
parser.add_argument('--final_noise_scale', type=float, default=0.3, metavar='G',
                    help='final noise scale (default: 0.3)')
parser.add_argument('--exploration_end', type=int, default=100, metavar='N',
                    help='number of episodes with noise (default: 100)')
parser.add_argument('--seed', type=int, default=4, metavar='N',
                    help='random seed (default: 4)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='batch size (default: 128)')
parser.add_argument('--num_steps', type=int, default=100000, metavar='N',
                    help='max episode length (default: 1000)')
parser.add_argument('--num_episodes', type=int, default=5000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')
args = parser.parse_args()
env = VehicleFollowingENV()
ATTACKER_LIMIT = env.ATTACKER_LIMIT
print("""
Environment Initializing...
The initial head car velocity is {}
The initial safe distance is     {}
The Nash Eq* Factor RC is        {}
""".format(env.v_head, env.d0, env.RC))
# writer = SummaryWriter()


ETA = 0.5


def fit_nash():
    agent_vehicle = NAF(args.gamma, args.tau, args.hidden_size,
                        env.observation_space, env.vehicle_action_space, mode='veh')
    agent_attacker = NAF(args.gamma, args.tau, args.hidden_size,
                         env.observation_space, env.attacker_action_space, mode='att')

    policy_vehicle = create_SL_model(env.observation_space, env.vehicle_action_space, mode='veh')
    policy_attacker = create_SL_model(env.observation_space, env.attacker_action_space, mode='att')

    memory_vehicle = ReplayMemory(1000000)
    memory_attacker = ReplayMemory(1000000)

    memory_SL_vehicle = ReplayMemory(100000)
    memory_SL_attacker = ReplayMemory(100000)

    ounoise_vehicle = OUNoise(env.vehicle_action_space) if args.ou_noise else None
    ounoise_attacker = OUNoise(env.attacker_action_space) if args.ou_noise else None

    param_noise_vehicle = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                                 desired_action_stddev=args.noise_scale,
                                                 adaptation_coefficient=1.05) if args.param_noise else None
    param_noise_attacker = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                                  desired_action_stddev=args.noise_scale,
                                                  adaptation_coefficient=1.05) if args.param_noise else None

    rewards = []
    eva_reward = []
    ave_reward = []
    tra_ac_veh = []
    tra_ac_att = []
    All_reward = []
    total_numsteps = 0
    updates = 0
    # while len(state_record) < 20:
    #     s, _, _ = env.step(*env.random_action())
    #     state_record.append(s)
    # print(torch.Tensor([state_record[-20:]]).shape)
    for i_episode in range(args.num_episodes):
        local_steps = 0
        state = env.reset()
        # state_record = [np.array([state])]
        state_record = [np.array([state])]
        episode_steps = 0
        while len(state_record) < 20:
            a, b = env.random_action()
            # s, _, _ = env.step(np.array([a]), np.array([b]))
            d, s, _, done = env.step()
            if done:
                s = np.array([env.reset()])
            state_record.append(s)
        if args.ou_noise:
            ounoise_vehicle.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                      i_episode) / args.exploration_end + args.final_noise_scale
            ounoise_vehicle.reset()
            ounoise_attacker.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                       i_episode) / args.exploration_end + args.final_noise_scale
            ounoise_attacker.reset()
        episode_reward = 0
        local_steps = 0
        while True:
            if random.random() < ETA:
                action_vehicle = agent_vehicle.select_action(torch.Tensor(state_record[-20:]), ounoise_vehicle,
                                                             param_noise_vehicle)[:, -1, :]
                action_attacker = agent_attacker.select_action(torch.Tensor(state_record[-20:]), ounoise_attacker,
                                                               param_noise_attacker)[:, -1, :]
            else:
                action_vehicle = torch.Tensor(
                    [policy_vehicle.predict(state_record[-1].reshape(-1, 4))])[0]
                action_attacker = torch.Tensor(
                    [policy_attacker.predict(state_record[-1].reshape(-1, 4))])[0]
            ac_v, ac_a = action_vehicle.numpy(), action_attacker.numpy()
            ac_v = ac_v / (sum(ac_v[0]) + 0.000000001)  # 4个权重和为1
            ac_a = ac_a / (sum(abs(ac_a[0])) + 0.000000001)  # 4个攻击绝对值和为1
            ac_a = np.array([[-0.25, 0, -0.75, 0]])
            _, next_state, reward, done = env.step(ac_v, ac_a)
            # print(ac_a, _)
            # print(done)
            state_record.append(next_state)
            local_steps += 1
            total_numsteps += 1
            episode_steps += 1
            episode_reward += reward
            memory_SL_vehicle.append(state_record[-1], ac_v)
            memory_SL_attacker.append(state_record[-1], ac_a)
            action_vehicle = torch.Tensor(action_vehicle)
            action_attacker = torch.Tensor(action_attacker)
            mask = torch.Tensor([not done])
            prev_state = torch.Tensor(state_record[-20:]).transpose(0, 1)
            next_state = torch.Tensor([next_state])
            reward_vehicle = torch.Tensor([reward])
            reward_attacker = torch.Tensor([env.RC - reward])
            memory_vehicle.push(prev_state, action_vehicle, mask, next_state, reward_vehicle)
            memory_attacker.push(prev_state, action_attacker, mask, next_state, reward_attacker)
            if done:
                rewards.append(episode_reward)
                if i_episode % 10 == 0:
                    print('Episode {} ends, local_steps {}. total_steps {}, instant ave-reward is {:.4f}'.format(
                        i_episode, local_steps, total_numsteps, episode_reward))
                break

        if len(memory_vehicle) > args.batch_size:  # 开始训练
            # print('begin training')
            for _ in range(args.updates_per_step):
                transitions_vehicle = memory_vehicle.sample(args.batch_size)
                batch_vehicle = Transition(*zip(*transitions_vehicle))

                transitions_attacker = memory_attacker.sample(args.batch_size)
                batch_attacker = Transition(*zip(*transitions_attacker))
                # print(batch_vehicle)

                trans_veh = memory_SL_vehicle.sample(args.batch_size)
                trans_att = memory_SL_attacker.sample(args.batch_size)

                states_veh = []
                actions_veh = []
                states_att = []
                actions_att = []
                for sample in trans_veh:
                    state_veh, act_veh = sample
                    states_veh.append(state_veh)
                    actions_veh.append(act_veh)
                for sample in trans_att:
                    state_att, act_att = sample
                    states_att.append(state_att)
                    actions_att.append(act_att)

                states_veh = np.reshape(states_veh, (-1, env.observation_space))
                states_att = np.reshape(states_att, (-1, env.observation_space))
                actions_veh = np.reshape(actions_veh, (-1, env.vehicle_action_space))
                actions_att = np.reshape(actions_att, (-1, env.attacker_action_space))

                policy_vehicle.fit(states_veh, actions_veh, verbose=False)
                policy_attacker.fit(states_att, actions_att, verbose=False)
                value_loss_vehicle, policy_loss_vehicle = agent_vehicle.update_parameters(batch_vehicle)
                value_loss_attacker, policy_loss_attacker = agent_attacker.update_parameters(batch_attacker)

                # writer.add_scalar('loss/value', value_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)

                updates += 1

        if i_episode % 10 == 0 and i_episode > 0:
            state = env.reset()
            state_record = [np.array([state])]
            eva_steps = 0
            while len(state_record) < 20:
                # a, b = env.random_action()
                # s, _, _ = env.step(np.array([a]), np.array([b]))
                _, s, _, _ = env.step()
                local_steps += 1
                state_record.append(s)
            evaluate_reward = 0
            while True:
                # la = np.random.randint(0, len(state_record) - 20, 1)[0]
                if random.random() < ETA:
                    action_vehicle = agent_vehicle.select_action(torch.Tensor(state_record[-20:]),
                                                                 ounoise_vehicle,
                                                                 param_noise_vehicle)[:, -1, :]
                    action_attacker = agent_attacker.select_action(torch.Tensor(state_record[-20:]),
                                                                   ounoise_attacker,
                                                                   param_noise_attacker)[:, -1, :]
                else:
                    action_vehicle = torch.Tensor([policy_vehicle.predict(
                        state_record[-1].reshape(-1, 4))])[0]
                    action_attacker = torch.Tensor([policy_attacker.predict(
                        state_record[-1].reshape(-1, 4))])[0]
                ac_v, ac_a = action_vehicle.numpy(), action_attacker.numpy()
                ac_v = ac_v / (sum(ac_v[0]) + 0.000000001)
                ac_a = ac_a / (sum(abs(ac_a[0])) + 0.000000001)
                ac_a = np.array([[-0.25, 0, -0.75, 0]])
                _, next_state, reward, done = env.step(ac_v, ac_a)
                tra_ac_veh.append(ac_v[0])
                tra_ac_att.append(ac_a[0])
                state_record.append(next_state)
                total_numsteps += 1
                eva_steps += 1
                local_steps += 1
                # print('eva_reward', reward)
                evaluate_reward += reward

                state = next_state[0]
                if done:
                    average_reward = np.mean(rewards[-10:])
                    print(
                        "{} % Episode finished, total numsteps: {}, eva-steps:{}, eva-reward: {}, average reward: {}".format(
                            i_episode / args.num_episodes * 100,
                            total_numsteps,
                            eva_steps,
                            evaluate_reward,
                            average_reward))
                    eva_reward.append(evaluate_reward)
                    ave_reward.append(average_reward)
                    # print(ac_v[0])
                    break
            # writer.add_scalar('reward/test', episode_reward, i_episode)
    env.close()
    df = pd.DataFrame()
    df['Reward'] = pd.Series(rewards)
    df['Eva'] = pd.Series(eva_reward)
    df['Tra'] = pd.Series(ave_reward)
    df2 = pd.DataFrame()
    df2['Weight'] = pd.Series(tra_ac_veh)
    df2['Attack'] = pd.Series(tra_ac_att)
    df.to_csv('./Result/reward_result_0607_max_attack_5000.csv', index=None)
    df2.to_csv('./Result/action_result_0607_max_attack_5000.csv', index=None)
    # np.savetxt('./Result/eva_result.csv', eva_reward, delimiter=',')
    # np.savetxt('./Result/ave_result.csv', ave_reward, delimiter=',')

    # f = plt.figure()
    # plt.plot(rewards[5:], label='Eva_reward')
    # plt.show()
    # AC_veh = np.array(tra_ac_veh)
    # AC_att = np.array(tra_ac_att)
    # print(AC_veh.shape)
    # print(AC_veh)
    # plt.plot(AC_veh[:, 0], label='Bacon1', alpha=0.2)
    # plt.plot(AC_veh[:, 1], label='Bacon2', alpha=0.2)
    # plt.plot(AC_veh[:, 2], label='Bacon3', alpha=0.2)
    # plt.plot(AC_veh[:, 3], label='Bacon4', alpha=0.2)
    # # plt.plot(ave_reward, label='Tra_ave_reward')
    # plt.legend()
    # plt.savefig('./Result/Veh_result_30.png', ppi=300)
    # plt.show()
    # print(AC_veh.shape)
    # print(AC_veh)
    # plt.plot(AC_att[:, 0], label='Attack1', alpha=0.2)
    # plt.plot(AC_att[:, 1], label='Attack2', alpha=0.2)
    # plt.plot(AC_att[:, 2], label='Attack3', alpha=0.2)
    # plt.plot(AC_att[:, 3], label='Attack4', alpha=0.2)
    # # plt.plot(ave_reward, label='Tra_ave_reward')
    # # plt.title('')
    # plt.legend()
    # plt.savefig('./Result/Att_result_0602.png', ppi=300)
    # plt.show()


if __name__ == '__main__':
    # main()
    fit_nash()
