import torch.nn as nn
from torch import squeeze, tanh, sigmoid, relu, log_softmax


class CNN(nn.Module):
    def __init__(self, activation_function, kernal_size, padding, pool, dilation):
        super(CNN, self).__init__()
        #2 conv layers
        self.conv1 = nn.Conv1d(6, 15, kernal_size, padding=padding, dilation=dilation).double()
        self.pool = nn.MaxPool1d(pool)
        self.conv2 = nn.Conv1d(15, 30, kernal_size, dilation=dilation).double()
        self.batch_norm1 = nn.BatchNorm1d(15).double()
        self.batch_norm2 = nn.BatchNorm1d(30).double()
        size = ((100 + 2 * padding + (1 - kernal_size) * max(dilation,1)) // pool + (1 - kernal_size) * max(dilation,1)) // pool
        self.fc1 = nn.Linear(30 * size, 128).double()
        self.fc2 = nn.Linear(128, 64).double()
        self.fc3 = nn.Linear(64, 26).double()
        self.a_func = activation_function
        self.size = size

        #3 conv layers
        # self.conv1 = nn.Conv1d(6, 12, kernal_size, padding=padding, dilation=dilation).double()
        # self.pool = nn.MaxPool1d(pool)
        # self.conv2 = nn.Conv1d(12, 20, kernal_size, dilation=dilation).double()
        # self.conv3 = nn.Conv1d(20, 36, kernal_size, dilation=dilation).double()
        # self.batch_norm1 = nn.BatchNorm1d(12).double()
        # self.batch_norm2 = nn.BatchNorm1d(20).double()
        # self.batch_norm3 = nn.BatchNorm1d(36).double()
        # size = (((100 + 2 * padding + (1 - kernal_size) * max(dilation, 1)) // pool + (1 - kernal_size) * max(dilation,1)) // pool + (1 - kernal_size) * max(dilation,1)) // pool
        # self.fc1 = nn.Linear(36 * size, 128).double()
        # self.fc2 = nn.Linear(128, 64).double()
        # self.fc3 = nn.Linear(64, 26).double()
        # self.a_func = activation_function
        # self.size = size
        # self.drop = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = eval("{}(x)".format(self.a_func))
        x = self.pool(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = eval("{}(x)".format(self.a_func))
        x = self.pool(x)
        # x = self.conv3(x)
        # x = self.batch_norm3(x)
        # x = eval("{}(x)".format(self.a_func))
        # x = self.pool(x)
        # x = x.view(-1, 36 * self.size)
        x = x.view(-1, 30 * self.size)
        x = self.fc1(x)
        x = eval("{}(x)".format(self.a_func))
        # x = self.drop(x)
        x = self.fc2(x)
        x = eval("{}(x)".format(self.a_func))
        x = log_softmax(self.fc3(x), dim=1)
        x = x.squeeze(1) # Flatten to [batch_size]
        return x