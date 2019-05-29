import numpy as np
from kalman import kalman_filter
import matplotlib.pyplot as plt
"""
    与原版相比，用step_km函数会在控制部分加了滤波，将附加量看成噪声
    reward在结束的时候返回更小的reward -10
    step最大值改成1000 
    
"""

SAMPLE_INTERVAL = 0.1
SPEED_LIMIT = 20
ACC_MODE = 0
UPPER_BOUND = 100
ATTACKER_LIMIT = np.array([1, 1, 1, 1])  # 攻击阈值
observation_space = 4
action_space = 4
vehicle_action_space = 4

"""
备份文件
"""


class VehicleFollowingENV(object):

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
        self.sensor_error = np.array([0.1, 0.1, 0.1, 0.1])  # 传感器误差高斯噪声
        self.lam = 0.02  # 控制量, reaction parameter
        self.d0 = np.random.randint(10, 30)  # 初始距离
        self.d = self.d0  # 实时距离
        self.init_v = np.random.random() * SPEED_LIMIT
        self.v_head = self.init_v  # 前车速度
        self.v = self.v_head  # 自车速度
        self.v_cal_raw = np.zeros(4)
        self.v_cal_raw_km = np.zeros(4)
        self.v_cal = 0
        self.T = 10
        self.epsilon = 0.01
        self.sample_interval = SAMPLE_INTERVAL  # 采样间隔
        self.a_head = 0  # 前车加速度
        self.action_car = 0  # 自身加速度
        self.step_number = 0
        self.observation_space = observation_space
        self.vehicle_action_space = vehicle_action_space
        self.attacker_action_space = vehicle_action_space
        self.RC = 0

    def reset(self):
        '''
        初始化环境, 返回未经攻击的前车速度的观测值
        :return
        v_head: 前车速度
        v:      自车速度
        '''
        self.d0 = np.random.randint(10, 30)  # 初始距离
        self.d = self.d0  # 实时距离
        self.init_v = np.random.random() * SPEED_LIMIT
        self.v_head = self.init_v  # 前车速度
        self.v = self.v_head  # 自车速度
        self.step_number = 0
        return self.v_cal_raw

    def control(self, action_weight=np.ones(4), action_attacker=np.zeros(4)):
        '''
        :param
        action_weight:权重
        action_attacker:攻击值

        :return:
        action_car  车辆速度
        '''
        # 权重归一化
        action_weight = action_weight / sum(action_weight)
        # print(action_weight, action_attacker)
        # 传感器随机误差
        SSerror = np.random.randn(4) * self.sensor_error
        # 更新前车原始数据
        self.v_cal_raw = self.v_head * np.ones(4) + action_weight * (SSerror + action_attacker)
        # print(self.v_cal_raw)
        # 前车的估计车速 公式7
        self.v_cal = self.v_head + np.sum(action_weight * (SSerror + action_attacker))

        # 控制结果 公式1
        self.action_car = self.lam * (self.v_cal - self.v)

    def km_control(self, action_weight=np.ones(4), action_attacker=np.zeros(4)):
        """
        将输出量加上km
        :param action_weight:
        :param action_attacker:
        :return:
        """
        # 权重归一化
        action_weight = action_weight / sum(action_weight)
        # print(action_weight, action_attacker)
        # 传感器随机误差
        SSerror = np.random.randn(4) * self.sensor_error
        # 更新前车原始数据
        self.v_cal_raw = self.v_head * np.ones(4) + action_weight * (SSerror + action_attacker)
        km1 = kalman_filter(0.001, 1)
        km2 = kalman_filter(0.001, 1)
        km3 = kalman_filter(0.001, 1)
        km4 = kalman_filter(0.001, 1)
        self.v_cal_raw_km[0] = km1.kalman(self.v_cal_raw[0])
        self.v_cal_raw_km[1] = km2.kalman(self.v_cal_raw[1])
        self.v_cal_raw_km[2] = km3.kalman(self.v_cal_raw[2])
        self.v_cal_raw_km[3] = km4.kalman(self.v_cal_raw[3])
        # print(self.v_cal_raw)
        # 前车的估计车速 公式7
        self.v_cal = np.sum(self.v_cal_raw_km)/4

        # 控制结果 公式1
        self.action_car = self.lam * (self.v_cal - self.v)
    def step(self, action_weight=np.ones(4),
             action_attacker=np.random.random(4)):  # =np.ones(4), action_attacker=np.zeros(4)):
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
        if self.d <= 1 or self.d >= UPPER_BOUND or self.step_number > 10000:
            is_done = True
        else:
            is_done = False
        # reward 用
        if (is_done):
            reward = -10
        else :
            reward = -(self.d - self.d0) ** 2 / 100 ** 2



        next_state = self.v_cal_raw
        return next_state, reward, is_done

    def step_km(self, action_weight=np.ones(4),
             action_attacker=np.random.random(4)):  # =np.ones(4), action_attacker=np.zeros(4)):
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
        # 更新控制
        self.km_control(action_weight, action_attacker)
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
        if self.d <= 1 or self.d >= UPPER_BOUND or self.step_number > 10000:
            is_done = True
        else:
            is_done = False
        # reward 用
        if (is_done):
            reward = -10
        else:
            reward = -(self.d - self.d0) ** 2 / 100 ** 2

        next_state = self.v_cal_raw
        return next_state, reward, is_done

    def random_action(self):
        weight = np.random.random(4)
        weight = weight / weight.sum()
        attrack = np.random.randn(4) + 1
        return weight, attrack

    def close(self):
        return


if __name__ == '__main__':
    env = VehicleFollowingENV()

    v_observe = env.reset()
    done = False
    eva_reward = []
    ave_reward = []
    episode = 500
    j=0
    while ( j < episode):


        j+=1
        v_observe = env.reset()
        done = False
        i = 0
        a = [env.v_head] * 1000
        adc = []
        vv = []
        rewards = []

        total_numsteps = 0
        updates = 0
        evaluate_reward = 0
        array = np.array(a)
        while (not done and i < 1000):
            i = i + 1
            weight = np.ones(4) / 4
            attrack = np.random.randn(4) + 1  # 攻击量取默认环境中的值，用卡尔曼滤波滤这个值
            next_state, reward, done = env.step(weight, attrack)   #不滤波结果
            # next_state, reward, done = env.step_km(weight, attrack) # 滤波结果
            evaluate_reward += reward
            #eva_reward.append(env.v_cal_raw[0] - env.v_head)
            # print('R({:d}):{:<6.2f},  Real Distance:{:.2f} m.   '.format(i, reward, env.d))
            # adc.append(env.v_cal_raw[0]-env.v_head)
            # vv.append(env.v_cal-env.v_head)
        eva_reward.append(evaluate_reward)
        average_reward = np.mean(eva_reward)
        ave_reward.append(average_reward)
        print('episode :{:d}  ave_reward:{:f}  evaluate_reward {:f}' .format(j,average_reward, evaluate_reward))
    plt.plot(eva_reward, label='Eva_reward')
    plt.plot(ave_reward, label='Tra_ave_reward')
    plt.show()


