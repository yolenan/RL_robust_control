from matplotlib import pyplot as plt
import re
import numpy as np
import pandas as pd
from DnsCarFollowENV2 import VehicleFollowingENV
import random

np.random.seed(1234)
random.seed(1234)


def plotReward(filename, step=10):
    reward = []
    episode = []
    with open(filename) as f:
        for line in f.readlines():
            if step != 10:
                res = re.findall('Episode (.*) ends, instant reward is (.*)', line)
            else:
                res = re.findall('Episode: (.*), total numsteps: .*, reward: .*, average reward: (.*)', line)
            if len(res) != 0:
                if float(res[0][1]) < -200000:
                    reward.append(-200000)

                else:
                    reward.append(float(res[0][1]))

                episode.append(float(res[0][0]))

    plt.xlabel('Episode')
    plt.ylabel('Reward')
    if step != 10:
        plt.title("Vehicle's total reward changes by 1 episode")
    else:
        plt.title("Vehicle's total reward changes by 10 episodes")
    plt.plot(episode, reward)
    # plt.ylim(0, 7000)
    plt.show()


def plotAction(filename, k):
    w1s = []
    w2s = []
    w3s = []
    w4s = []
    step = []
    i = 0

    with open(filename) as f:
        for line in f.readlines():
            res = re.findall('\[\s*(\S*)\s*(\S*)\s*(\S*)\s*(\S*)\s*\]', line)
            if len(res) != 0:
                (w1, w2, w3, w4) = res[0]
                w1s.append(float(w1))
                w2s.append(float(w2))
                w3s.append(float(w3))
                w4s.append(float(w4))
                step.append(i)
                i += 1

    i = 0
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    while i < len(step):
        if "weight" in filename:
            total = 1
            # total = np.average(w1s[i:i+k]) + np.average(w2s[i:i+k]) + np.average(w3s[i:i+k]) + np.average(w4s[i:i+k])
        else:
            total = 1
        y1.append(np.average(w1s[i:i + k]) / total)
        y2.append(np.average(w2s[i:i + k]) / total)
        y3.append(np.average(w3s[i:i + k]) / total)
        y4.append(np.average(w4s[i:i + k]) / total)
        i += k
    plt.xlabel('Total step/' + str(k))
    if "weight" in filename:
        plt.ylim(0, 1)
        plt.ylabel('Weight')
        plt.title('Weight changes with total step')
        plt.legend(['w1', 'w2', 'w3', 'w4'])


    else:
        plt.ylabel('Attack')
        plt.title('Attack changes with total step')
        plt.legend(['a1', 'a2', 'a3', 'a4'])

    plt.step(range(len(y1)), y1, c='r')
    plt.step(range(len(y1)), y2, c='k')
    plt.step(range(len(y1)), y3, c='b')
    plt.step(range(len(y1)), y4, c='g')

    plt.show()


def plotDistance(filename, k):
    dis = []
    i = 0
    legends = []
    with open(filename) as f:
        for line in f.readlines():
            res = re.findall('The distance is (.*)', line)
            if len(res) == 0:
                i += 1
                dis.append([])
                legends.append('episode' + str(i))
            else:
                dis[-1].append(float(res[0]))

    for dis_data in dis[9::10]:
        plt.step(range(len(dis_data)), dis_data)
    plt.legend(legends[9::10])
    plt.xlabel('Number step in episode.')
    plt.ylabel('Distance')
    plt.title('Distance changes within one episode')
    plt.show()


def parseActionCSV(filename, action, k):
    data = pd.read_csv(filename)
    name = action
    if name != 'Eva_distance':
        data['w1'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[0]))
        data['w2'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[1]))
        try:
            data['w3'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[2]))
        except:
            print(data[action].apply(lambda x: x.replace('[', '').replace(']', '').split()))
        data['w4'] = data[action].apply(lambda x: float(x.replace('[', '').replace(']', '').split()[3]))
        w1 = data.w1.values
        w2 = data.w2.values
        w3 = data.w3.values
        w4 = data.w4.values
        y1 = []
        y2 = []
        y3 = []
        y4 = []
        i = 0
        total = 1
        while i < w1.shape[0]:
            y1.append(np.average(w1[i:i + k]) / total)
            y2.append(np.average(w2[i:i + k]) / total)
            y3.append(np.average(w3[i:i + k]) / total)
            y4.append(np.average(w4[i:i + k]) / total)
            i += k
        # plt.subplots(1, figsize=(10, 8))
        plt.xlabel('Total step/' + str(k), fontsize=12)
        if action == 'Weight':
            plt.ylim(min(y1 + y2 + y3 + y4) * 0.99, max(y1 + y2 + y3 + y4) * 1.01)
        plt.ylabel(action, fontsize=12)
        plt.title(action + ' changes with total step', fontsize=12)
        plt.step(range(len(y1)), y1, c='r')
        plt.step(range(len(y1)), y2, c='k')
        plt.step(range(len(y1)), y3, c='b')
        plt.step(range(len(y1)), y4, c='g')
        if action == 'Weight':
            plt.legend(['w1', 'w2', 'w3', 'w4'])
        else:
            plt.legend(['a1', 'a2', 'a3', 'a4'])

        # plt.show()
    else:
        data[action] = data[action].astype('float32')
        w = [[]]
        legends = []
        i = 0
        count = 0
        for distance in data[action].values:
            count += 1
            if distance < 20 or distance > 30 or count > 2000:
                w[-1].append(distance)
                w.append([])
                legends.append('episode' + str(i))
                count = 0
                i += 1
            else:
                w[-1].append(distance)
        print(i)
        print(legends[0::len(legends) // 10])
        for dis_data in w[0::len(w) // 10]:
            plt.step(range(len(dis_data)), dis_data)
        plt.legend(legends[0::len(legends) // 10], fontsize=8)
        plt.xlabel('Number step in episode.', fontsize=12)
        plt.ylabel('Distance', fontsize=12)
        plt.title('Distance changes within one episode', fontsize=12)
        # plt.show()


def subspace(x):
    x = re.sub('\[\s+', '[', x)
    x = re.sub('\s+\]', ']', x)
    x = re.sub('\s+', ',', x)
    return x


def ResolveDistance(filename, k):
    env = VehicleFollowingENV()
    env.reset()
    data = pd.read_csv(filename)
    data['Weight_array'] = data['Weight'].apply(lambda x: eval('np.array(' + subspace(x) + ')'))
    data['Attack_array'] = data['Attack'].apply(lambda x: eval('np.array(' + subspace(x) + ')'))
    i = 0
    weights = data['Weight_array'].values
    attacks = data['Attack_array'].values
    distances = [[]]
    distance_data = []
    legends = []
    done_count = 0
    while i < data['Weight_array'].shape[0]:
        ac_v = np.array([weights[i]])
        ac_a = np.array([attacks[i]])
        d, state, _, is_done = env.step(ac_v, ac_a)
        distance_data.append(d)
        if is_done:
            distances[-1].append(d)
            distances.append([])
            done_count += 1
            print(done_count)
            legends.append('episode' + str(done_count))
            env.reset()
        else:
            distances[-1].append(d)
        i += 1
    data['Eva_distance'] = pd.Series(distance_data)
    data[['Weight', 'Attack', 'Eva_distance']].to_csv(filename, index=False)
    for dis_data in distances[0::len(distances) // 10]:
        plt.step(range(len(dis_data)), dis_data)
    plt.legend(legends[0::len(legends) // 10], fontsize=8)
    plt.xlabel('Number step in episode.', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Distance changes within one episode', fontsize=12)


filename = 'action_result_061304_4beacon_RC0_1000_eva_eta'
actionfile = './Result/' + filename + '.csv'
rewardfile = './Result/' + ''
k = int(re.findall('_(\d*000)[_\.]', actionfile)[0])
print(k)


def main():
    plt.figure(figsize=(18, 6))
    # plt.subplot(1,3,1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.text(0.05, 0.5,'_'.join(actionfile.replace('bacon', 'beacon').split('_')[4:7])[:-4]+'Episode', fontsize=16)
    #
    # ax = plt.gca()
    # ax.axes.get_yaxis().set_visible(False)
    # ax.axes.get_xaxis().set_visible(False)
    # ax.spines['top'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    try:
        plt.subplot(1, 3, 1)
        parseActionCSV(actionfile, 'Eva_distance', k)
    except:
        ResolveDistance(actionfile, k)
    plt.subplot(1, 3, 2)
    parseActionCSV(actionfile, 'Eva_Weight', k//10)
    plt.subplot(1, 3, 3)
    parseActionCSV(actionfile, 'Eva_Attack', k//10)


if __name__ == '__main__':
    main()
    # ResolveDistance(actionfile, k)
    # parseActionCSV(actionfile, 'Eva_distance', k)
    # plotReward('rewardNash_True_RC_0_AttackMode_1_RewardMode_3.txt', 10)
    # plotAction('attacker_actionNash_True_RC_0_AttackMode_1_RewardMode_3.txt', 20000)
    # plotAction('vehicle_weightNash_True_RC_0_AttackMode_1_RewardMode_3.txt', 20000)
    # plotDistance('DistanceNash_True_RC_0_AttackMode_1_RewardMode_3.txt', 0)
    plt.savefig('./Result/figure_plot/'+filename + '.png', dpi=300)
    plt.show()
