import numpy as np
import os


def get_data(filename):
    return np.loadtxt(filename, delimiter=",", usecols=(1, 2, 3, 4, 5, 6))


data = []
label = []
os.chdir('./unnamed_train_data')
students = os.listdir('.')
for student in students:
    os.chdir(student)
    files = os.listdir('.')
    for file in files:
        data.append(get_data(file))
        label.append(file[0])
    os.chdir('..')
os.chdir('..')
np.save('instances.npy', np.array(data))
np.save('labels.npy', np.array(label))
