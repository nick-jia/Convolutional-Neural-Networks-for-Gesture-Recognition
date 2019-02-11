import numpy as np
from sklearn.model_selection import train_test_split

data = np.load('instances.npy')
labels = np.load('labels.npy')

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=0)

np.save('train_data.npy', X_train)
np.save('val_data.npy', X_test)
np.save('train_label.npy', y_train)
np.save('val_label.npy', y_test)