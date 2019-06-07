import numpy as np
from math import *
from scipy.optimize import minimize

SAMPLE_INTERVAL = 1
SPEED_LIMIT = 20
ACC_MODE = 0
UPPER_BOUND = 5
ATTACKER_LIMIT = np.array([1, 1, 1, 1])  # 攻击阈值
observation_space = 4
action_space = 4
vehicle_action_space = 4
VMAX = 20
import matplotlib.pyplot as plt

VMIN = 0
AMAX = 5

"""
备份文件
"""


class VehicleFollowingENV(object):
    ATTACKER_LIMIT = np.array([1, 1, 1, 1])

    def __init__(self):
        '''
        :param
        d0:     初始车距, 范围在10m~30m间
        d:      当前车距
        v_Head: 前车速度, 范围在0-20m/s之间
        v:      自车速度, 初始值等于前车速度
        v_cal    前车测量速度
        sample_interval: 采样时间, 单位: s
        a_head: 前车加速度
        '''
        self.sensor_error = np.array([0, 0, 0, 0])  # 传感器误差高斯噪声
        self.lam = 1  # 控制量, reaction parameter
        # self.d0 = np.random.randint(10, 30)  # 初始距离
        self.d0 = 25
        self.d = self.d0  # 实时距离
        self.init_v = np.random.random() * SPEED_LIMIT
        self.v_head = self.init_v  # 前车速度
        self.v = self.v_head  # 自车速度
        self.v_cal_raw = np.zeros(4)
        self.v_cal = 0
        self.sample_interval = SAMPLE_INTERVAL  # 采样间隔
        self.a_head = 0  # 前车加速度
        self.action_car = 0  # 自身加速度
        self.step_number = 0
        self.observation_space = observation_space
        self.vehicle_action_space = vehicle_action_space
        self.attacker_action_space = vehicle_action_space
        self.RC = 0
        self.reward_mode = 3
        self.defend_mode = 0
        self.attack_mode = 3

    def reset(self):
        '''
        初始化环境, 返回未经攻击的前车速度的观测值
        :return
        v_head: 前车速度
        v:      自车速度
        '''
        self.d = self.d0
        self.step_number = 0
        self.v_cal_raw = self.init_v * np.ones(4)
        return self.v_cal_raw

    def func(self, args):
        fun = lambda x: abs(np.dot(x, args))
        return fun

    def con(self, args):
        cons = ({'type': 'eq', 'fun': lambda x: np.dot(x, args) - 1})
        return cons

    def control(self, action_weight=np.ones(4), action_attacker=np.zeros(4)):
        '''
        :param
        action_weight:权重
        action_attacker:攻击值

        :return:
        action_car  车辆速度
        '''
        # 传感器随机误差
        # SSerror = np.random.randn(4) * self.sensor_error
        SSerror = np.zeros(4)  # 假设没有传感器误差
        # 更新前车原始数据
        self.v_cal_raw = self.v_head * np.ones(4) + action_weight * (SSerror + action_attacker)
        # 前车的估计车速 公式7
        if self.defend_mode == 1:
            args = action_attacker[0]
            args1 = np.ones(4)
            x0 = np.array([0.25, 0.25, 0.25, 0.25])
            cons = self.con(args1)
            res = minimize(self.func(args), x0, bounds=((0, None), (0, None), (0, None), (0, None)), method='SLSQP',
                           constraints=cons)
            print(res.x)
            action_weight = np.array([res.x])
        self.v_cal = self.v_head + np.sum(action_weight * (SSerror + action_attacker))
        # 控制结果 公式1
        self.action_car = self.lam * (self.v_cal - self.v)

    def step(self, action_weight=np.array([np.ones(4)]),
             action_attacker=np.array([np.zeros(4)])):  # =np.ones(4), action_attacker=np.zeros(4)):
        '''
        环境的步进, 输入攻击者和自车的权重动作，通过控制器, 返回新的Reward和观测值
        :param
        action_car: 车辆的加速度
        action_attacker: 插入的假数据

        :return
        next_state: 经过伪造的数据
        reward:     距离与安全距离的差平方
        is_done:    是否发生碰撞
        '''
        # 更新步数
        self.step_number = self.step_number + 1
        # 在环境中限制Attacker
        # print(action_attacker)
        [a0, a1, a2, a3] = action_attacker[0]
        # a0 = 1 if a0 > 1 else a0
        # a0 = -1 if a0 < -1 else a0
        # a1 = 1 if a1 > 1 else a1
        # a1 = -1 if a1 < -1 else a1
        # a2 = 0.5 if a2 > 0.5 else a2
        # a2 = -0.5 if a2 < -0.5 else a2
        # a3 = 1.5 if a3 > 1.5 else a3
        # a3 = -1.5 if a3 < -1.5 else a3
        if self.attack_mode == 0:
            # 攻击a1
            action_attacker = np.array([[a0, 0, 0, 0]])
        elif self.attack_mode == 1:
            # 攻击a2和a3
            action_attacker = np.array([[0, 0, a2, a3]])
        else:
            # 全部攻击
            action_attacker = np.array([[a0, a1, a2, a3]])
        # 更新控制
        self.control(action_weight, action_attacker)
        # 前车行驶距离(保留没有变)
        s_head = self.v_head * self.sample_interval + 0.5 * self.a_head * self.sample_interval * self.sample_interval
        # 自车行驶距离（保留没有变）
        s_self = self.v * self.sample_interval + 0.5 * self.action_car * self.sample_interval * self.sample_interval
        # 新的车距（保留没有变）
        self.d = self.d + s_head - s_self
        # 更新车速
        self.v_head = self.v_head + self.a_head * self.sample_interval
        self.v = self.v + self.action_car * self.sample_interval
        # 返回结果
        if abs(self.d-self.d0) > UPPER_BOUND or self.step_number > 100000000:
            is_done = True
        else:
            is_done = False
        # reward 用
        if self.reward_mode == 0:
            if (is_done):
                reward = -10
            else:
                reward = -(self.d - self.d0) ** 2 / 100 ** 2
        elif self.reward_mode == 1:
            if abs(self.d - self.d0) < 1:
                reward = 1
            elif done:
                reward = -1
            else:
                reward = 1 / abs(self.d - self.d0) * 10 ** 0
        elif self.reward_mode == 2:
            factor = np.array([0.2, 0.3, 0.5])
            r_v = log(100 * (self.v - VMIN) / (VMAX - VMIN) + 0.99, (VMAX - VMIN) / 2) - 1
            r_a = (self.v_cal - self.v) / AMAX
            if self.d0 < self.d < 30:
                r_y = 1
            elif 0 <= self.d <= self.d0:
                r_y = -10
            elif self.d >= 30:
                r_y = -10
            reward = (np.array([r_v, r_a, r_y]) * factor).sum()
        elif self.reward_mode == 3:
            delta_d = abs(self.d - self.d0)
            reward = (delta_d - UPPER_BOUND) ** 2 / (UPPER_BOUND ** 2)

        next_state = self.v_cal_raw
        # print(action_weight, action_attacker)
        return self.d, next_state, reward, is_done

    def random_action(self):
        weight = np.random.random(4)
        weight = weight / weight.sum()
        attack = np.random.randn(4) * ATTACKER_LIMIT
        return np.array([weight]), np.array([attack])

    def close(self):
        return


if __name__ == '__main__':
    env = VehicleFollowingENV()

    # v_observe = env.reset()
    done = False

    i = 0
    rewards = []
    distance = []
    done_count = 0
    while (i < 10000):
        i = i + 1
        d, next_state, reward, done = env.step(*env.random_action())
        # d, next_state, reward, done = env.step()
        rewards.append(reward)
        distance.append(d)
        if done:
            v_observe = env.reset()
            done = False
            print('Episode reward {}'.format(sum(rewards)))
            rewards = []
            # break
            done_count += 1
        # print(next_state)
        # print('R({:d}):{:<6.2f},  Real Distance:{:.2f} m.  Done:{} '.format(i, reward, env.d, done))
    print(done_count)
    f = plt.figure()
    plt.plot(rewards, label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.legend()
    plt.show()
    # plt.figure()
    plt.plot(distance, label='Distance')
    plt.xlabel('Step')
    plt.ylabel('Distance/m')
    plt.legend()
    plt.show()
    print('Done num', done_count)
