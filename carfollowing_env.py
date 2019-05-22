import numpy as np

SAMPLE_INTERVAL = 3
SPEED_LIMIT = 60
ACC_MODE = 0


class VehicleFollowingENV(object):
    def __init__(self):
        '''
        :param
        d0:     初始车距, 范围在10m~30m间
        v_Head: 前车速度, 范围在0-60km/h之间
        v:      自车速度, 初始值等于前车速度
        sample_interval: 采样时间, 单位: s
        a_head: 前车加速度
        '''
        self.d0 = np.random.randint(10, 30)
        self.v_head = np.random.random() * SPEED_LIMIT
        self.v = self.v_head
        self.sample_interval = SAMPLE_INTERVAL
        self.a_head = 0
        self.d = self.d0

    def close(self):
        pass

    def reset(self):
        '''
        初始化环境, 返回未经攻击的前车速度的观测值和自身速度
        :return
        v_head: 前车速度
        v:      自车速度
        '''
        self.v_head = np.random.random() * SPEED_LIMIT
        self.v = self.v_head
        self.d = self.d0
        return [self.v_head] * 4 + [self.v]

    def step(self, action_car, action_attacker=np.zeros(5)):
        '''
        环境的步进, 输入攻击者和自车的动作, 返回新的Reward和观测值

        :param
        action_car: 车辆的加速度
        action_attacker: 插入的假数据

        :return
        next_state: 经过伪造的数据
        reward:     距离与安全距离的差平方
        is_done:    是否发生碰撞
        '''
        # 前车行驶距离
        s_head = self.v_head * self.sample_interval + 0.5 * self.a_head * self.sample_interval * self.sample_interval
        # 自车行驶距离
        s_self = self.v * self.sample_interval + 0.5 * action_car * self.sample_interval * self.sample_interval
        # 新的车距
        self.d = self.d + s_head - s_self
        # 更新车速
        self.v_head = self.v_head + self.a_head * self.sample_interval
        self.v = self.v + action_car * self.sample_interval
        # 返回结果
        if self.d <= 0:
            is_done = True
        else:
            is_done = False

        reward = (self.d - self.d0) ** 2

        next_state = [self.v_head] * 4 + [self.v] + action_attacker
        # next_state.append(self.v)

        print(next_state)

        return reward, next_state, is_done


# t_state = [self.v_head] * 4 + action_attacker
#
#         return reward, next_state, is_done


if __name__ == '__main__':

    env = VehicleFollowingENV()

    v_observe = env.reset()
    print(v_observe)
    done = False
    weight = np.random.random(4)
    weight = weight / weight.sum()
    while not done:
        action = np.random.random() / 100
        reward, next_state, done = env.step(action)
        print('R(n):{:<6.2f},  Real Distance:{:.2f} m.'.format(reward, env.d))
