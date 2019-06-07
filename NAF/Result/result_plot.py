import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('reward_result_0602.csv')
plt.plot(df['Eva'],label='Evaluate')
plt.plot(df['Tra'],label='Train')
plt.legend()
plt.show()