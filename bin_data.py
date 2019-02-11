import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = np.load('instances.npy')
labels = np.load('labels.npy')
df = pd.DataFrame({'Parameters': ['ax', 'ay', 'az', 'wx', 'wy', 'wz']})

for label in np.unique(labels):
    temp = data[labels == label]
    avg = [np.average(temp[:, :, i]) for i in range(6)]
    std = [np.std(temp[:, :, i]) for i in range(6)]
    df['{}_avg'.format(label)] = avg
    df['{}_std'.format(label)] = std

df.to_csv('stats.csv')

for gesture in ['a', 'b', 'c']:
    fig, ax = plt.subplots()
    ax.bar(np.arange(6), df['{}_avg'.format(gesture)], yerr=df['{}_std'.format(gesture)])
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(df['Parameters'])
    ax.set_title('Basic Statistics of gesture {}'.format(gesture))
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig('Basic Statistics of gesture {}'.format(gesture))
    plt.show()
