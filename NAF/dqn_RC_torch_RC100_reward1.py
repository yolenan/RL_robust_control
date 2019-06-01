#!/usr/bin/env python
import torch
from NAF.naf import NAF
from tensorboardX import pytorch_graph
from NAF.replay_memory import ReplayMemory, Transition
import numpy as np
import random
from NAF.ENV_TODO2 import VehicleFollowingENV
from ounoise import OUNoise
from NAF.Supervised_Learning import create_SL_model
from param_noise import AdaptiveParamNoiseSpec, ddpg_distance_metric
import argparse
import os
from matplotlib import pyplot as plt
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

torch.manual_seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.001, metavar='G',
                    help='discount factor for model (default: 0.001)')
parser.add_argument('--ou_noise', type=bool, default=True)
parser.add_argument('--param_noise', type=bool, default=False)
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
parser.add_argument('--num_episodes', type=int, default=1000, metavar='N',
                    help='number of episodes (default: 1000)')
parser.add_argument('--hidden_size', type=int, default=128, metavar='N',
                    help='number of episodes (default: 128)')
parser.add_argument('--updates_per_step', type=int, default=5, metavar='N',
                    help='model updates per simulator step (default: 5)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 1000000)')

env = VehicleFollowingENV()
args = parser.parse_args()
print("""
Environment Initializing...
The initial head car velocity is {}
The initial safe distance is     {}
The Nash Eq* Factor RC is        {}
The Reward Calculation Mode is   {}
The Attack Mode is               {}
""".format(env.v_head, env.d0, env.RC, env.reward_mode, env.attack_mode))
# writer = SummaryWriter()


ETA = 0.5

def fit_nash():
    reward_file = open('reward_RC100_r1.txt', 'w')
    attack_file = open('attacker_action_RC100_r1.txt', 'w')
    weight_file = open('vehicle_weight_RC100_r1.txt', 'w')
    distance_file = open('Distance_RC100_r1.txt', 'w')

    reward_file.write("""
Environment Initializing...
The initial head car velocity is {}
The initial safe distance is     {}
The Nash Eq* Factor RC is        {}
The Reward Calculation Mode is   {}
The Attack Mode is               {}
""".format(env.v_head, env.d0, env.RC, env.reward_mode, env.attack_mode))

    agent_vehicle = NAF(args.gamma, args.tau, args.hidden_size,
                        env.observation_space, env.vehicle_action_space)
    agent_attacker = NAF(args.gamma, args.tau, args.hidden_size,
                         env.observation_space, env.attacker_action_space)


    policy_vehicle = create_SL_model(env.observation_space, env.vehicle_action_space)
    policy_attacker = create_SL_model(env.observation_space, env.attacker_action_space)

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
    total_numsteps = 0
    updates = 0
    for i_episode in range(args.num_episodes):
        state = env.reset()
        if args.ou_noise:
            ounoise_vehicle.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                              i_episode) / args.exploration_end + args.final_noise_scale
            ounoise_vehicle.reset()

            ounoise_attacker.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                      i_episode) / args.exploration_end + args.final_noise_scale
            ounoise_attacker.reset()
        episode_reward = 0
        while True:
            if random.random() < ETA:
                action_vehicle = agent_vehicle.select_action(torch.Tensor([[state]]), ounoise_vehicle, param_noise_vehicle)
                action_attacker = agent_attacker.select_action(torch.Tensor([[state]]), ounoise_attacker, param_noise_attacker)
            else:
                action_vehicle = torch.Tensor([policy_vehicle.predict(state.reshape(-1, 4))/policy_vehicle.predict(state.reshape(-1, 4)).sum()])
                action_attacker = torch.Tensor([policy_attacker.predict(state.reshape(-1, 4))/policy_attacker.predict(state.reshape(-1, 4)).sum()])

            attack_file.write(str(action_attacker.numpy()[0])+'\n')
            weight_file.write(str(action_vehicle.numpy()[0])+ '\n')

            next_state, reward, done = env.step(action_vehicle.numpy()[0], action_attacker.numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            memory_SL_vehicle.append(state, action_vehicle.numpy()[0])
            memory_SL_attacker.append(state, action_attacker.numpy()[0])

            action_vehicle = torch.Tensor(action_vehicle)
            action_attacker = torch.Tensor(action_attacker)

            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])

            reward_vehicle = torch.Tensor([reward])
            reward_attacker = torch.Tensor([env.RC - reward])

            memory_vehicle.push(torch.Tensor([[state]]), action_vehicle, mask, next_state, reward_vehicle)
            memory_attacker.push(torch.Tensor([[state]]), action_attacker, mask, next_state, reward_attacker)

            state = next_state.numpy()[0][0]

            if done:
                rewards.append(episode_reward)
                if i_episode % 10:
                    print('Episode {} ends, instant reward is {:.2f}'.format(i_episode, episode_reward))
                    reward_file.write('Episode {} ends, instant reward is {:.2f}\n'.format(i_episode, episode_reward))
                break

        if len(memory_vehicle) > args.batch_size:  # 开始训练
            for _ in range(args.updates_per_step):
                transitions_vehicle = memory_vehicle.sample(args.batch_size)
                batch_vehicle = Transition(*zip(*transitions_vehicle))

                transitions_attacker = memory_attacker.sample(args.batch_size)
                batch_attacker = Transition(*zip(*transitions_attacker))

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
                actions_veh =np.reshape(actions_veh, (-1, env.vehicle_action_space))
                actions_att =np.reshape(actions_att, (-1, env.attacker_action_space))

                policy_vehicle.fit(states_veh, actions_veh, verbose=False)
                policy_attacker.fit(states_att, actions_att, verbose=False)
                value_loss_vehicle, policy_loss_vehicle = agent_vehicle.update_parameters(batch_vehicle)
                value_loss_attacker, policy_loss_attacker = agent_attacker.update_parameters(batch_attacker)

                # writer.add_scalar('loss/value', value_loss, updates)
                # writer.add_scalar('loss/policy', policy_loss, updates)

                updates += 1

        if i_episode % 10 == 0 and i_episode != 0:
            distance_file.write('{} episode starts, recording distance...\n'.format(i_episode))
            state = env.reset()
            evaluate_reward = 0
            while True:
                if random.random() < ETA:
                    action_vehicle = agent_vehicle.select_action(torch.Tensor([[state]]), ounoise_vehicle,
                                                                 param_noise_vehicle)
                    action_attacker = agent_attacker.select_action(torch.Tensor([[state]]), ounoise_attacker,
                                                                   param_noise_attacker)
                else:
                    action_vehicle = torch.Tensor([policy_vehicle.predict(
                        state.reshape(-1, 4)) / policy_vehicle.predict(state.reshape(-1, 4)).sum()])
                    action_attacker = torch.Tensor([policy_attacker.predict(
                        state.reshape(-1, 4)) / policy_attacker.predict(state.reshape(-1, 4)).sum()])

                next_state, reward, done = env.step(action_vehicle.numpy()[0], action_attacker.numpy()[0])
                distance_file.write('The distance is '+str(env.d)+'\n')
                evaluate_reward += reward


                state = next_state[0]
                if done:
                    print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode,
                                                                                                   total_numsteps,
                                                                                                   evaluate_reward,
                                                                                                   np.mean(rewards[-10:])))
                    reward_file.write("Episode: {}, total numsteps: {}, reward: {}, average reward: {}\n".format(i_episode,
                                                                                                   total_numsteps,
                                                                                                   evaluate_reward,
                                                                                                   np.mean(rewards[-10:])))
                    break
                # writer.add_scalar('reward/test', episode_reward, i_episode)


    env.close()
    reward_file.close()
    attack_file.close()
    weight_file.close()


def main():
    agent_vehicle = NAF(args.gamma, args.tau, args.hidden_size,
                         env.observation_space, env.vehicle_action_space)
    agent_attacker = NAF(args.gamma, args.tau, args.hidden_size,
                         env.observation_space, env.attacker_action_space)


    vehicle_memory = ReplayMemory(1000000)
    attacker_memory = ReplayMemory(1000000)


    vehicle_ounoise = OUNoise(env.vehicle_action_space) if args.ou_noise else None
    attacker_ounoise = OUNoise(env.attacker_action_space) if args.ou_noise else None

    param_noise_vehicle = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                         desired_action_stddev=args.noise_scale,
                                         adaptation_coefficient=1.05) if args.param_noise else None
    param_noise_attacker = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                         desired_action_stddev=args.noise_scale,
                                         adaptation_coefficient=1.05) if args.param_noise else None

    rewards = []
    total_numsteps = 0
    updates = 0

    for i_episode in range(args.num_episodes):
        state = torch.Tensor([[env.reset()]])  # 4-dimensional velocity observation

        if args.ou_noise:
            vehicle_ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                              i_episode) / args.exploration_end + args.final_noise_scale
            vehicle_ounoise.reset()

            attacker_ounoise.scale = (args.noise_scale - args.final_noise_scale) * max(0, args.exploration_end -
                                                                                      i_episode) / args.exploration_end + args.final_noise_scale
            attacker_ounoise.reset()

        episode_reward = 0

        while True:
            action_vehicle = agent_vehicle.select_action(state, vehicle_ounoise, param_noise_vehicle)
            action_attacker = agent_attacker.select_action(state, attacker_ounoise, param_noise_attacker)

            next_state, reward, done = env.step(action_vehicle.numpy()[0], action_attacker.numpy()[0])
            total_numsteps += 1
            episode_reward += reward

            action_vehicle = torch.Tensor(action_vehicle)
            action_attacker = torch.Tensor(action_attacker)

            mask = torch.Tensor([not done])
            next_state = torch.Tensor([next_state])

            reward_vehicle = torch.Tensor([-reward])
            reward_attacker = torch.Tensor([env.RC+reward])

            vehicle_memory.push(state, action_vehicle, mask, next_state, reward_vehicle)
            attacker_memory.push(state, action_attacker, mask, next_state, reward_attacker)

            state = next_state

            if len(vehicle_memory) > args.batch_size:
                for _ in range(args.updates_per_step):
                    transitions_vehicle = vehicle_memory.sample(args.batch_size)
                    batch_vehicle = Transition(*zip(*transitions_vehicle))

                    transition_attacker = attacker_memory.sample(args.batch_size)
                    batch_attacker = Transition(*zip(*transition_attacker))

                    value_loss_1, policy_loss_1 = agent_vehicle.update_parameters(batch_vehicle)
                    value_loss_2, policy_loss_2 = agent_attacker.update_parameters(batch_attacker)

                    # writer.add_scalar('loss/value', value_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)

                    updates += 1

            if done:
                break

        # writer.add_scalar('reward/train', episode_reward, i_episode)

        # Update param_noise based on distance metric
        t = args.batch_size
        if args.param_noise:
            episode_transitions_vehicle = vehicle_memory.memory[vehicle_memory.position - t:vehicle_memory.position]
            states_vehicle = torch.cat([transition[0] for transition in episode_transitions_vehicle], 0)
            unperturbed_actions_vehicle = agent_vehicle.select_action(states_vehicle, None, None)
            perturbed_actions_vehicle = torch.cat([transition[1] for transition in episode_transitions_vehicle], 0)

            ddpg_dist_vehicle = ddpg_distance_metric(perturbed_actions_vehicle.numpy(), unperturbed_actions_vehicle.numpy())
            param_noise_vehicle.adapt(ddpg_dist_vehicle)

            episode_transitions_attacker = attacker_memory.memory[attacker_memory.position - t:attacker_memory.position]
            states_attacker = torch.cat([transition[0] for transition in episode_transitions_attacker], 0)
            unperturbed_actions_attacker = agent_attacker.select_action(states_attacker, None, None)
            perturbed_actions_attacker = torch.cat([transition[1] for transition in episode_transitions_attacker], 0)

            ddpg_dist_attacker = ddpg_distance_metric(perturbed_actions_attacker.numpy(), unperturbed_actions_attacker.numpy())
            param_noise_attacker.adapt(ddpg_dist_attacker)

        rewards.append(episode_reward)

        if i_episode % 10 == 0:
            state = torch.Tensor([[env.reset()]])
            episode_reward = 0
            while True:
                action_vehicle = agent_vehicle.select_action(state, vehicle_ounoise, param_noise_vehicle)
                action_attacker = agent_attacker.select_action(state, attacker_ounoise, param_noise_attacker)

                next_state, reward, done = env.step(action_vehicle.numpy()[0], action_attacker.numpy()[0])
                episode_reward += reward

                next_state = torch.Tensor([[next_state]])
                state = next_state
                if done:
                    break

            # writer.add_scalar('reward/test', episode_reward, i_episode)

            rewards.append(episode_reward)
            print("Episode: {}, total numsteps: {}, reward: {}, average reward: {}".format(i_episode, total_numsteps,
                                                                                           rewards[-1],
                                                                                           np.mean(rewards[-10:])))

    env.close()

def visualization():
    agent_vehicle = NAF(args.gamma, args.tau, args.hidden_size,
                        env.observation_space, env.vehicle_action_space)
    agent_attacker = NAF(args.gamma, args.tau, args.hidden_size,
                         env.observation_space, env.attacker_action_space)

    state = env.reset()
    ounoise_vehicle = OUNoise(env.vehicle_action_space) if args.ou_noise else None
    ounoise_attacker = OUNoise(env.attacker_action_space) if args.ou_noise else None

    param_noise_vehicle = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                                 desired_action_stddev=args.noise_scale,
                                                 adaptation_coefficient=1.05) if args.param_noise else None
    param_noise_attacker = AdaptiveParamNoiseSpec(initial_stddev=0.05,
                                                  desired_action_stddev=args.noise_scale,
                                                  adaptation_coefficient=1.05) if args.param_noise else None

    action_vehicle = agent_vehicle.select_action(torch.Tensor([[state]]), ounoise_vehicle, param_noise_vehicle)
    action_attacker = agent_attacker.select_action(torch.Tensor([[state]]), ounoise_attacker, param_noise_attacker)

    pytorch_graph.graph(agent_vehicle, (action_vehicle, action_attacker))

if __name__ == '__main__':
    # main()
    fit_nash()
    # visualization()