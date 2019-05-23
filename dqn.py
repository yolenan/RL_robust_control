"""Main DQN agent."""
from objectives import mean_huber_loss
import random
import numpy as np
from policy import GreedyEpsilonPolicy, LinearDecayGreedyEpsilonPolicy, UniformRandomPolicy
from core import ReplayMemory, Preprocessor
from DnsCarFollowENV2 import VehicleFollowingENV
from copy import deepcopy
from dqn_RC import create_SL_model
from keras.metrics import categorical_crossentropy, mse

# 用于创建SL网络的输入输出维度
DIM_STATES = 4
NUM_ACTIONS = 32
RC = 0  # 纳什均衡的合作-竞争参数 R1 + R2 = RC
ETA = 0.5  # NFSP选择最优策略和平均策略的概率

class DQNAgent:
    """Class implementing DQN.

    This is a basic outline of the functions/parameters you will need
    in order to implement the DQNAgnet. This is just to get you
    started. You may need to tweak the parameters, add new ones, etc.


    Parameters
    ----------
    q_network: keras.models.Model
      Your Q-network model.
    preprocessor: deeprl_hw2.core.Preprocessor
      The preprocessor class. See the associated classes for more
      details.
    memory: deeprl_hw2.core.Memory
      Your replay memory.
    gamma: float
      Discount factor.
    target_update_freq: float
      Frequency to update the target network. You can either provide a
      number representing a soft target update (see utils.py) or a
      hard target update (see utils.py and Atari paper.)
    num_burn_in: int
      Before you begin updating the Q-network your replay memory has
      to be filled up with some number of samples. This number says
      how many.
    train_freq: int
      How often you actually update your Q-Network. Sometimes
      stability is improved if you collect a couple samples for your
      replay memory, for every Q-network update that you run.
    batch_size: int
      How many samples in each minibatch.
    """

    def __init__(self,
                 q_network,
                 q_network2,
                 preprocessor: Preprocessor(),
                 RLmemory: ReplayMemory(),
                 SLmemory: ReplayMemory(),
                 policy,
                 gamma,
                 target_update_freq,
                 num_burn_in,
                 train_freq,
                 batch_size,
                 algorithm='DoubuleDQN',
                 render=False):
        self.net = q_network
        self.net2 = q_network2
        self.pre = preprocessor
        self.rl_mem = RLmemory
        self.sl_mem = SLmemory
        self.policy = policy
        self.gamma = gamma
        self.renew = target_update_freq
        self.burn_in = num_burn_in
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.algorithm = algorithm
        self.render = render

    def compile(self, optimizer, loss_func):
        """Setup all of the TF graph variables/ops.

        This is inspired by the compile method on the
        keras.models.Model class.

        This is a good place to create the target network, setup your
        loss function and any placeholders you might need.
        
        You should use the mean_huber_loss function as your
        loss_function. You can also experiment with MSE and other
        losses.

        The optimizer can be whatever class you want. We used the
        keras.optimizers.Optimizer class. Specifically the Adam
        optimizer.
        """
        self.net.compile(optimizer=optimizer, loss=loss_func)
        self.net2.compile(optimizer=optimizer, loss=loss_func)

    def calc_q_values(self, state, net):
        """Given a state (or batch of states) calculate the Q-values.

        Basically run your network on these states.

        Return
        ------
        Q-values for the state(s)
        """
        # with tf.Session() as f:
        #     print(state.eval())

        q_value = net.predict(state, steps=32)
        return q_value

    def select_action(self, state, net, process='training'):
        """Select the action based on the current state.

        You will probably want to vary your behavior here based on
        which stage of training your in. For example, if you're still
        collecting random samples you might want to use a
        UniformRandomPolicy.

        If you're testing, you might want to use a GreedyEpsilonPolicy
        with a low epsilon.

        If you're training, you might want to use the
        LinearDecayGreedyEpsilonPolicy.

        This would also be a good place to call
        process_state_for_network in your preprocessor.

        Returns
        --------
        selected action
        """
        assert process in ['sampling', 'testing', 'training'], 'Unsupported process.'

        epsilon = 0.1
        start_value = 1
        end_value = 0.1
        num_steps = 10 ** 6

        q_values = self.calc_q_values(state, net)

        if process == 'sampling':
            action = UniformRandomPolicy(len(q_values)).select_action()
        elif process == 'testing':
            action = GreedyEpsilonPolicy(epsilon).select_action(q_values)
        else:
            action = LinearDecayGreedyEpsilonPolicy(start_value, end_value, num_steps).select_action(q_values)

        return action

    def fit(self, env, num_iterations, max_episode_length=None):
        """Fit your model to the provided environment.

        Its a good idea to print out things like loss, average reward,
        Q-values, etc to see if your agent is actually improving.

        You should probably also periodically save your network
        weights and any other useful info.

        This is where you should sample actions from your network,
        collect experience samples and add them to your replay memory,
        and update your network parameters.

        Parameters
        ----------
        env: VehicleFollowingEnv
          This is your Atari environment. You should wrap the
          environment using the wrap_atari_env function in the
          utils.py
        num_iterations: int
          How many samples/updates to perform.
        max_episode_length: int
          How long a single episode should last before the agent
          resets. Can help exploration.
        """
        # state = self.pre.process_state_for_memory(env.reset()) #函数内容是pass
        state = env.reset()  # 获取初始状态
        tmp = 0
        prev_action = np.zeros(4)  # 初始前操作
        states = [state]
        state_ = np.zeros(4)
        for i in range(num_iterations):
            # env.render()
            if max_episode_length and i > max_episode_length:
                break
            if state_.all() <= 0:
                action = np.random.random(4)  # 初始状态产生随机权重
            else:
                # state_ = tf.squeeze(state_)
                # state_ = tf.reshape(state_, 1)
                action = self.select_action(state_, process='testing')
            # print(action)
            # print('action', action)
            next_state, reward, done = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(i + 1))
                break
            # next_state = self.pre.process_state_for_memory(next_state)
            states.append(next_state)
            tmp += 1
            self.mem.append(state, prev_action, reward, next_state, done)
            # if tmp >= 6:
            #     # frames = states[-5:-1]
            #     # frames2 = states[-4:]
            #     # state_ = tf.concat([tf.expand_dims(i, 2) for i in frames], 2)
            #     # next_state_ = tf.concat([tf.expand_dims(i, 2) for i in frames2], 2)
            #     print(state, next_state)
            #
            #     states = states[-5:]
            prev_action = action
            if i % self.renew == 0 and i != 0:
                self.net2 = self.net
            if i != 0 and i % self.train_freq == 0:
                print('{}th iteration, {}th train starts.'.format(i, i // self.train_freq))
                batches = min(self.batch_size, len(self.mem))
                current_states = []
                q_values = []
                for samples in self.mem.sample(batches):
                    current_state, action, reward, next_state, is_done = [samples.state,
                                                                          samples.action,
                                                                          samples.reward,
                                                                          samples.next_state,
                                                                          samples.done]
                    # state = tf.reshape(tf.squeeze(current_state), 4)
                    # next_state = tf.reshape((tf.squeeze(current_state)), 4)
                    current_states.append(state)
                    target = reward
                    if not is_done:
                        if self.algorithm == 'NDQN':
                            target = reward + self.gamma * np.amax(self.net2.predict(next_state, steps=32)[0])
                        elif self.algorithm == 'DQN':
                            target = reward + self.gamma * np.amax(self.net.predict(next_state, steps=32)[0])
                        elif self.algorithm == 'DoubleDQN':
                            target = reward
                            # TODO
                        elif self.algorithm == 'DuelingDQN':
                            target = reward
                    target_f = self.net.predict(state, steps=10)
                    print(len(target_f))
                    print(action)
                    target_f[action] = target
                    q_values.append(target_f)
                # current_states = tf.reshape(current_states, 4)
                q_values = np.reshape((q_values), (-1, 6))
                print(current_states.shape, q_values.shape)
                self.net.fit(current_states, q_values, steps_per_epoch=self.batch_size)

    def fit_nash(self, env: VehicleFollowingENV, num_iterations, episode, total_step, max_episode_length=None):
        """
        Fit with Nash Equilibrium
        """
        # RL network: LSTM
        self.p1_net = self.net  # target network
        self.p1_net2 = self.net2

        self.p2_net = deepcopy(self.net)
        self.p2_net2 = deepcopy(self.net2)

        # SL network: NN
        self.p1_policy = create_SL_model(DIM_STATES, NUM_ACTIONS)
        self.p2_policy = create_SL_model(DIM_STATES, NUM_ACTIONS)

        self.p1_policy.compile('Adam', categorical_crossentropy)
        self.p2_policy.compile('Adam', mse)

        # ReplayMemory
        self.p1_RL_mem = ReplayMemory(max_size=100000)
        self.p2_RL_mem = ReplayMemory(max_size=100000)
        self.p1_SL_mem = ReplayMemory(max_size=100000)
        self.p2_SL_mem = ReplayMemory(max_size=100000)

        # MainLoop
        state = env.reset()
        total_reward = 0
        done = False

        for i in num_iterations:
            total_step += 1

            # if self.render:
            #     env.render()
            if max_episode_length and i > max_episode_length:
                break

            if np.random.random() < ETA:
                best_response = True
            else:
                best_response = False

            if best_response:
                p1_action = self.select_action(state, net=self.p1_net)
                p2_action = self.select_action(state, net=self.p2_net)
            else:
                p1_action = self.select_action(state, net=self.p1_policy)
                p2_action = self.select_action(state, net=self.p2_policy)

            next_state, reward, done = env.step(action_weight=p1_action, action_attacker=p2_action)

            self.p1_RL_mem.append((state, p1_action, RC-reward, next_state, done))
            self.p2_RL_mem.append((state, p2_action, reward, next_state, done))
            self.p1_SL_mem.append((state, p1_action))
            self.p2_SL_mem.append((state, p2_action))

            total_reward += reward

            if done:
                with open(self.algorithm + 'total_reward.txt', 'a') as f:
                    f.write('Episode ({}), reward: ({})\n'.format(episode, total_reward))
                print("Episode finished after {} time steps, total_reward is {}...".format(i, total_reward))
                break

            if total_step % self.renew == 0 and total_step != 0:
                self.p1_net2 = self.p1_net
                self.p2_net2 = self.p2_net

            # if total_step % 100000 == 0:
            #     self.save(total_step)

            if total_step >= self.burn_in and total_step % self.train_freq == 0:
                batches = min(self.batch_size, len(self.p1_RL_mem))
                p1_states, p1_actions, p1_q_values = self.sample_from_Replay_Memory(batches, self.p1_RL_mem, self.p1_net)
                p2_states, p2_actions, p2_q_values = self.sample_from_Replay_Memory(batches, self.p2_RL_mem, self.p2_net)

                self.p1_net.fit(p1_states, p1_q_values)
                self.p2_net.fit(p2_states, p2_q_values)
                self.p1_policy.fit(p1_states, p1_actions)
                self.p2_policy.fit(p2_states, p2_actions)

            state = next_state
        return total_step, done


    def sample_from_Replay_Memory(self, batches, ReplayMemory, Net):
        current_states = []
        actions = []
        q_values = []

        for samples in ReplayMemory.sample(batches):
            state, action, reward, next_state, is_done = [samples.state,
                                                          samples.action,
                                                          samples.reward,
                                                          samples.next_state,
                                                          samples.done]

            next_state = np.expand_dims(np.asarray(next_state).astype(np.float64), axis=0)

            current_states.append(state)
            actions.append(action)
            target = reward
            if not is_done:
                target = reward + self.gamma * np.amax(Net.predict(next_state)[0])
            target_f = Net.predict(state)[0]
            target_f[action] = target
            q_values.append(target_f)
        current_states = np.reshape(current_states, (-1, DIM_STATES))
        q_values = np.reshape((q_values), (-1, NUM_ACTIONS))
        return current_states, actions, q_values

    def evaluate(self, env, num_episodes, max_episode_length=None):
        """Test your agent with a provided environment.
        
        You shouldn't update your network parameters here. Also if you
        have any layers that vary in behavior between train/test time
        (such as dropout or batch norm), you should set them to test.

        Basically run your policy on the environment and collect stats
        like cumulative reward, average episode length, etc.

        You can also call the render function here if you want to
        visually inspect your policy.
        """
        for i in range(num_episodes):
            total = 0
            state = np.zeros(4)
            tmp = 0
            prev_action = 0
            states = [state]
            state_ = -1
            while True:
                if max_episode_length and i > max_episode_length:
                    break
                if state_ == -1:
                    action = np.random.randint(6)
                else:
                    action = self.select_action(state_)
                next_state, reward, done, _ = env.step(action)
                if tmp < 6:
                    # next_state = self.pre.process_state_for_memory(next_state)
                    states.append(next_state)
                    tmp += 1
                if tmp >= 6:
                    # frames = states[-5:-1]
                    # frames2 = states[-4:]
                    # state_ = tf.concat([tf.expand_dims(i, 2) for i in frames], 2)
                    # next_state_ = tf.concat([tf.expand_dims(i, 2) for i in frames2], 2)
                    self.mem.append(state, prev_action, reward, next_state, done)
                    states.append(state)
                    states = states[-5:]
                prev_action = action
                state = next_state
                total += reward
            print('Episode {}, total reward is {}'.format(i, total))
