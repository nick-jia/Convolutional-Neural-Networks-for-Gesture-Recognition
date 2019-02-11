import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.utils.data as data
import json
from model import CNN

test_data = np.load('test_data.npy')
test_data = np.array([i.transpose() for i in test_data])

with open('config.json') as file:
        config = json.load(file)

batch_size = config['batch_size']
learning_rate = config['learning_rate']
num_epochs = config['epoch']
optm = config['optimizer']
afunc = config['activation function']
kernal_size = config['kernal_size']
padding = config['padding']
pool = config['pool']
dilation = config['dilation']


class TestDataset(data.Dataset):

    def __init__(self, X):
        X = torch.from_numpy(X)
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        sample = (self.X[index])
        return sample


test_data = TestDataset(test_data)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
model = CNN(afunc, kernal_size, padding, pool, dilation)
model = torch.load('model.pt')

preds = np.array([], dtype=np.float)

for i, data in enumerate(test_loader, 0):
    inputs = data
    outputs = model(inputs).squeeze()
    preds = np.concatenate((preds, torch.max(outputs, 1)[1]))

# np.savetxt('predictions.csv', preds, fmt='%.1f')
np.savetxt('predictions.txt', preds)