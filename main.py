from time import time
import matplotlib.pyplot as plt
import json
import random
import os

import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold

import torch
from torch.utils.data import DataLoader

from model import CNN
from dataset import GestureDataset

#set random seed
seed = 0
np.random.seed(seed)

#read data files
data = np.load('instances.npy')
# data = np.load('./data/normalized_data.npy')
# data = np.delete(data, (2), axis=2) #drop z
# data = data[:,:,0:2] #only x,y
X = np.array([i.transpose() for i in data])
labels = np.load('labels.npy')

#One-hot code the labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(labels)

oneh_encoder = OneHotEncoder()
y =  oneh_encoder.fit_transform(y.reshape(-1, 1)).toarray()

#define functions
def load_data(X_train, X_test, y_train, y_test, batch_size):

    train_dataset = GestureDataset(X_train, y_train)
    test_dataset = GestureDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def load_model(lr, optm, afunc, kernal_size, padding, pool, dilation):

    model = CNN(afunc, kernal_size, padding, pool, dilation)
    loss_fnc = torch.nn.CrossEntropyLoss()
    if optm == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr = lr, momentum=0.5)
    elif optm == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr = lr, eps=1e-8)#, weight_decay=0.001)

    return model, loss_fnc, optimizer


def evaluate(model, val_loader, loss_fnc):
    total_val_loss = 0.0
    total_val_err = 0.0
    total_epoch = 0

    for i, data in enumerate(val_loader, 0):
        inputs, labels = data
        outputs = model(inputs).squeeze()
        loss = loss_fnc(outputs, torch.max(labels, 1)[1])
        for i in range(outputs.shape[0]):
            outputs[i] = outputs[i] == max(outputs[i])
        corr = outputs != labels
        total_val_err += int(corr.sum()/2)
        total_val_loss += loss.item()
        total_epoch += len(labels)

    return float(total_val_err) / total_epoch, float(total_val_loss) / total_epoch


def plot_res(epochs, train, val, model_num, seed):
    plt.figure()
    plt.plot(range(epochs), train, label='Train')
    plt.plot(range(epochs), val, label='Validation')
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.title("model_num={}_seed{}".format(model_num, seed))
    # plt.show(block=False)
    plt.savefig("model{}_seed{}.png".format(model_num, seed))


def save_config(bs, lr, epoch, optm, afunc, val_accuracy, model_num, seed, comment, kernal_size, padding, pool, dilation, fold, th_fold):
    data = {}
    data['batch_size'] = bs
    data['learning_rate'] = lr
    data['epoch'] = epoch
    data['optimizer'] = optm
    data['activation function'] = afunc
    data['kernal_size'] = kernal_size
    data['padding'] = padding
    data['pool'] = pool
    data['dilation'] = dilation
    data['comment'] = comment
    data['fold'] = '{}/{}'.format(th_fold,fold)

    with open('model{}_seed{}_accuracy{}.json'.format(model_num, seed, val_accuracy), 'w') as outfile:
        json.dump(data, outfile)


def main():
    global X,y
    save = True
    model_num = 16
    learning_rate, batch_size, num_epochs = 0.0005, 64, 100
    optm = "Adam"
    afunc = "relu"
    comment = 'Batch normalization twice, dilation = 2 twice, not normalized data'
    kernal_size = 10
    padding = 5
    pool = 2
    dilation = 2
    fold = 1

    if save:
        if not os.path.exists('/model{}'.format(model_num)):
            os.makedirs('./model{}'.format(model_num))
        if not 'model{}'.format(model_num) in os.getcwd():
            os.chdir('./model{}'.format(model_num))

    for i in range(5): #the number in parenthesis = number of different random seed tried
        seed = round(random.random()*1000)
        torch.manual_seed(seed)

        # split data

        if fold == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=seed)
            th_fold = 1
            # save split data
            np.save('train_data.npy', X_train)
            np.save('val_data.npy', X_test)
            np.save('train_label.npy', y_train)
            np.save('val_label.npy', y_test)
            for i in range(1):
        #uncomment the section below for cross validation

        #         pass
        #
        # else:
        #     kf = KFold(n_splits=fold, random_state=seed)
        #     th_fold = 0
        #     for train_index, test_index in kf.split(X):
        #         # print("TRAIN:", train_index, "TEST:", test_index)
        #         X_train, X_test = X[train_index], X[test_index]
        #         y_train, y_test = y[train_index], y[test_index]
        #         th_fold += 1

                train_loader, val_loader = load_data(X_train, X_test, y_train, y_test, batch_size)
                model, loss_fnc, optimizer = load_model(learning_rate, optm, afunc, kernal_size, padding, pool, dilation)

                train_err = np.ones(num_epochs)
                train_loss = np.ones(num_epochs)
                val_err = np.ones(num_epochs)
                val_loss = np.ones(num_epochs)

                start_time = time()

                for epoch in range(num_epochs):  # loop over the dataset multiple times
                    total_train_loss = 0.0
                    total_train_err = 0.0

                    total_epoch = 0

                    for i, data in enumerate(train_loader, 0):
                        inputs, labels = data
                        optimizer.zero_grad()
                        outputs = model(inputs).squeeze()
                        loss = loss_fnc(outputs, torch.max(labels, 1)[1])
                        loss.backward()
                        optimizer.step()

                        for i in range(outputs.shape[0]):
                            outputs[i] = outputs[i] == max(outputs[i])
                        corr = outputs != labels
                        total_train_err += int(corr.sum()/2)
                        total_train_loss += loss.item()
                        total_epoch += len(labels)

                    train_err[epoch] = float(total_train_err) / total_epoch
                    train_loss[epoch] = float(total_train_loss) / (i+1)
                    val_err[epoch], val_loss[epoch] = evaluate(model, val_loader, loss_fnc)
                    if val_err[epoch] == min(val_err):
                        best_model, best_epoch, best_accuracy = model, epoch, 1 - val_err[epoch]

                    print("Epoch {}: Train err: {}, Train loss: {} | Validation err: {}, Validation loss: {}".format(epoch + 1, train_err[epoch], train_loss[epoch], val_err[epoch], val_loss[epoch]))

                print('Finished Training, best accuracy is {} at {} epoch.'.format(best_accuracy, best_epoch+1))
                end_time = time()
                elapsed_time = end_time - start_time
                print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
                if save:
                    plot_res(num_epochs, 1 - train_err, 1 - val_err, model_num, seed)
                    torch.save(model, 'model{}_seed{}_accuracy{}.pt'.format(model_num, seed, best_accuracy))
                    save_config(batch_size, learning_rate, num_epochs, optm, afunc, best_accuracy, model_num, seed, comment, kernal_size, padding, pool, dilation, fold, th_fold)


if __name__ == "__main__":
    main()
