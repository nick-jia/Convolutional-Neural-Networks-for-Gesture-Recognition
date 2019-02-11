import numpy as np
import matplotlib.pyplot as plt


def get_data(filename):
    return np.loadtxt(filename, delimiter=",", usecols=(1, 2, 3, 4, 5, 6))


def plot_gesture(gesture):
    parameters = ['ax', 'ay', 'az', 'wx', 'wy', 'wz']
    x = [i for i in range(0, len(gesture))]
    plt.figure()
    for i in range(0, len(gesture[0])):
        plt.plot(x, [gesture[j][i] for j in range(0, len(gesture))], label=parameters[i])
    plt.legend(loc='best')
    plt.show()


a0 = get_data('./unnamed_train_data/student0/a_3.csv')
plot_gesture(a0)
a1 = get_data('./unnamed_train_data/student1/a_3.csv')
plot_gesture(a1)
a2 = get_data('./unnamed_train_data/student3/a_3.csv')
plot_gesture(a2)
b0 = get_data('./unnamed_train_data/student0/b_3.csv')
plot_gesture(b0)
b1 = get_data('./unnamed_train_data/student1/b_3.csv')
plot_gesture(b1)
b2 = get_data('./unnamed_train_data/student3/b_3.csv')
plot_gesture(b2)
