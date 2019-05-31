from matplotlib import pyplot as plt
import re
import numpy as np

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
            res = re.findall('\[\[\s*(\S*)\s*(\S*)\s*(\S*)\s*(\S*)\s*\]\]', line)
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
            total = np.average(w1s[i:i+k]) + np.average(w2s[i:i+k]) + np.average(w3s[i:i+k]) + np.average(w4s[i:i+k])
        else:
            total = 1
        y1.append(np.average(w1s[i:i+k])/total)
        y2.append(np.average(w2s[i:i+k])/total)
        y3.append(np.average(w3s[i:i+k])/total)
        y4.append(np.average(w4s[i:i+k])/total)
        i += k
    plt.xlabel('Total step')
    if "weight" in filename:
        # plt.ylim(-1,1)
        plt.ylabel('Weight')
        plt.title('Weight changes with total step')
        plt.legend(['w1', 'w2', 'w3', 'w4'])


    else:
        plt.ylabel('Attack')
        plt.title('Attack changes with total step')
        plt.legend(['a1','a2','a3','a4'])

    plt.step(range(len(y1)), y1, c='r')
    plt.step(range(len(y1)), y2, c='k')
    plt.step(range(len(y1)), y3, c='b')
    plt.step(range(len(y1)), y4, c='g')

    if "weight" in filename:
        plt.legend(['w1', 'w2', 'w3', 'w4'])
    else:
        plt.legend(['a1','a2','a3','a4'])

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
                legends.append('episode'+str(i*10))
            else:
                dis[-1].append(float(res[0]))

    for dis_data  in dis[9::10]:
        plt.step(range(len(dis_data)), dis_data)
    plt.legend(legends[9::10])
    plt.xlabel('Number step in episode.')
    plt.ylabel('Distance')
    plt.title('Distance changes within one episode')
    plt.show()


plotAction('attacker_action_RC100_r1.txt', 20000)
# plotReward('reward_RC100_r1.txt', 1)
# plotDistance('Distance_RC100_r1.txt', 0)