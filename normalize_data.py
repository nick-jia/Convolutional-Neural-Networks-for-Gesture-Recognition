import numpy as np
import os

data = np.load('instances.npy')
normalized_data = np.zeros(data.shape)
for i in range(0, data.shape[0]):
    for j in range(0, data.shape[2]):
        avg = np.average(data[i, :, j])
        std = np.std(data[i, :, j])
        for k in range(0, data.shape[1]):
            normalized_data[i, k, j] = (data[i, k, j] - avg) / std
if not os.path.exists('data'):
    os.makedirs('data')
np.save('./data/normalized_data.npy', normalized_data)
