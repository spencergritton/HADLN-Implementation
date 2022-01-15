import pandas as pd
import numpy as np

# Import MIT BIH data
train_x_mit = pd.read_csv("data/mitbih_train.csv").to_numpy()
test_x_mit = pd.read_csv("data/mitbih_test.csv").to_numpy()

train_x = train_x_mit
test_x = test_x_mit

# Separate y from train and test
train_y = train_x[:, -1].squeeze()
test_y = test_x[:, -1].squeeze()

# Delete y from train and test
train_x = np.delete(train_x, -1, axis=1)
test_x = np.delete(test_x, -1, axis=1)

# Normalize train and test
mean = np.mean(train_x, axis=0)
std = np.std(train_x, axis=0)

train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

train_x = np.expand_dims(train_x, -1)
train_y = np.expand_dims(train_y, -1)
test_x = np.expand_dims(test_x, -1)
test_y = np.expand_dims(test_y, -1)

# One hot encode outputs to allow for weighted cross entropy loss
train_y_onehot = []
test_y_onehot = []
for i in range(train_y.shape[0]):
	y = train_y[i]
	if y == 0: train_y_onehot.append([1, 0, 0, 0, 0])
	elif y == 1: train_y_onehot.append([0, 1, 0, 0, 0])
	elif y == 2: train_y_onehot.append([0, 0, 1, 0, 0])
	elif y == 3: train_y_onehot.append([0, 0, 0, 1, 0])
	else: train_y_onehot.append([0, 0, 0, 0, 1])

for i in range(test_y.shape[0]):
	y = test_y[i]
	if y == 0: test_y_onehot.append([1, 0, 0, 0, 0])
	elif y == 1: test_y_onehot.append([0, 1, 0, 0, 0])
	elif y == 2: test_y_onehot.append([0, 0, 1, 0, 0])
	elif y == 3: test_y_onehot.append([0, 0, 0, 1, 0])
	else: test_y_onehot.append([0, 0, 0, 0, 1])

# Save files to data folder
np.save("data/train_x", train_x)
np.save("data/train_y", np.array(train_y_onehot))
np.save("data/test_x", test_x)
np.save("data/test_y", np.array(test_y_onehot))
