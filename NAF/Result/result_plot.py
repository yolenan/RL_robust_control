import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('ggplot')
df = pd.read_csv('reward_result_0607.csv')
# plt.plot(df['Eva'],label='Evaluate')
plt.plot(df['Reward'][1:], label='Train')
plt.legend()
plt.xlabel('Episode')
plt.ylabel('Episode reward')
plt.title('Adversary Learning Result 4 Bacons')
plt.show()
