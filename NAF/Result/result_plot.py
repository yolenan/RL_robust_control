import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')


def plot_result(mode='result', title='Adversary Learning Result'):
    df = pd.read_csv('reward_result_0608_2bacon_RC0_1000.csv')
    # df = pd.read_csv('reward_result_0608_4bacon_5000.csv')
    # df2 = pd.read_csv('reward_result_0608_4bacon_RC50_1000.csv')
    # df3 = pd.read_csv('reward_result_0608_4bacon_RC100_1000.csv')
    if mode == 'result':
        plt.plot(df['Reward'][1:], label='Train')
    elif mode == 'evaluate':
        plt.plot(df['Eva'], label='2bacon-Evaluate-RC0')
        plt.plot(df['Tra'], label='2bacon-Train-RC0')
        # plt.plot(df2['Eva'], label='4bacon-Evaluate-RC50')
        # plt.plot(df2['Tra'], label='4bacon-Train-RC50')
        # plt.plot(df3['Eva'], label='4bacon-Evaluate-RC100')
        # plt.plot(df3['Tra'], label='4bacon-Train-RC100')
    plt.legend()
    plt.xlabel('Episode')
    plt.ylabel('Episode reward')
    plt.title(title)
    plt.show()


def plot_action(mode='veh'):
    AC_veh = pd.read_csv('action_result_0607_max_attack_5000.csv')['Weight'].values
    print(AC_veh[0][0])
    if mode == 'veh':
        plt.plot(AC_veh[:, 0], label='Bacon1', alpha=0.2)
        plt.plot(AC_veh[:, 1], label='Bacon2', alpha=0.2)
        plt.plot(AC_veh[:, 2], label='Bacon3', alpha=0.2)
        plt.plot(AC_veh[:, 3], label='Bacon4', alpha=0.2)
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Value')
    # plt.title(title)
    plt.show()


plot_result(mode='evaluate')
# plot_result()
# plot_action()