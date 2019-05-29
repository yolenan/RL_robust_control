import numpy as np
import matplotlib.pyplot as plt

#  因为都是观测速度的 所以观测矩阵和转移矩阵都是1 省略
class kalman_filter:
    def __init__(self, Q, R):
        self.Q = Q      # 状态转移噪声协防差
        self.R = R      #  测量噪声协防差

        self.P_k_k1 = 1  # 协防差
        self.Kg = 0   # 卡尔曼增益
        self.P_k1_k1 = 1  # 更新后协防差
        self.x_k_k1 = 0   # 前一时刻最优结果
        self.ADC_OLD_Value = 0
        self.Z_k = 0
        self.kalman_adc_old = 0  # 卡尔曼预测结果

    def kalman(self, ADC_Value):

        self.Z_k = ADC_Value
        if (self.kalman_adc_old == 0):
            self.kalman_adc_old =  ADC_Value
        if (abs(self.kalman_adc_old - ADC_Value) >= 60):  # 随便加的阈值
            self.x_k1_k1 = ADC_Value * 0.382 + self.kalman_adc_old * 0.618
        else:
            self.x_k1_k1 = self.kalman_adc_old;

        self.x_k_k1 = self.x_k1_k1  # 状态转移预测值
        self.P_k_k1 = self.P_k1_k1 + self.Q  # 新的协防差

        self.Kg = self.P_k_k1 / (self.P_k_k1 + self.R)

        kalman_adc = self.x_k_k1 + self.Kg * (self.Z_k - self.kalman_adc_old)
        self.P_k1_k1 = (1 - self.Kg) * self.P_k_k1
        self.P_k_k1 = self.P_k1_k1

        self.kalman_adc_old = kalman_adc

        return kalman_adc


if __name__ == '__main__':
    kalman_filter1 = kalman_filter(0.001, 1)
    kalman_filter2 = kalman_filter(0.001, 1)
    a = [100] * 200
    b=[0]*200
    array = np.array(a)
    array2=np.array(b)
    attrack = np.random.randn(1) + 1
    s = np.random.normal(0, 15, 200)
    s2= np.random.normal(1,1,200)
    test_array = array + s
    test_array2 = array2 + s2
    adc = []
    test2=[]
    for i in range(200):
        adc.append(kalman_filter1.kalman(test_array[i]))
        test2.append(kalman_filter2.kalman(test_array2[i]))
    # plt.plot(adc)
    # plt.plot(array)
    #plt.plot(test_array)
    plt.plot(test2)
    plt.plot(test_array2)
    plt.show()

