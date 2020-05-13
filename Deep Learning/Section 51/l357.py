import numpy as np 
from sklearn import preprocessing

raw_data = np.loadtxt('Audiobooks_data.csv', delimiter=',')

raw_train = raw_data[:, 1:-1]
raw_test = raw_data[:,-1]

# shuffling
shuffled_indx = np.arange(raw_data.shape[0])
np.random.shuffle(shuffled_indx)
print(shuffled_indx)

shuffled_train = raw_train[shuffled_indx]
shuffled_test = raw_test[shuffled_indx]

#balancing
num_ones = sum(shuffled_test)

count_zeros_seen = 0
to_be_deleted_indx = []

for i in range(shuffled_test.shape[0]):
	if shuffled_test[i] == 0:
		count_zeros_seen += 1
		if count_zeros_seen > num_ones:
			to_be_deleted_indx.append(i)

balanced_inputs = np.delete(shuffled_train, to_be_deleted_indx, axis=0)
balanced_targets = np.delete(shuffled_test, to_be_deleted_indx, axis=0)

# data is not shuffled anymore. So we shuffle again
shuffled_indx = np.arange(balanced_inputs.shape[0])
np.random.shuffle(shuffled_indx)

balanced_inputs = balanced_inputs[shuffled_indx]
balanced_targets = balanced_targets[shuffled_indx]

print(balanced_targets)

# scaling
scaled_inputs = preprocessing.scale(balanced_inputs)

# dividing into train, validation, test
division = [0.8, 0.1]
num_all_inputs = scaled_inputs.shape[0]
num_train = int (division[0] * num_all_inputs)
num_validation = int (division[1] * num_all_inputs)
num_test = num_all_inputs - num_train - num_validation

final_train_inputs = scaled_inputs[: num_train]
final_train_targets = balanced_targets[: num_train]

final_validation_inputs = scaled_inputs[num_train : num_train+num_validation]
final_validation_targets = balanced_targets[num_train : num_train+num_validation]

final_test_inputs = scaled_inputs[num_train+num_validation :]
final_test_targets = balanced_targets[num_train+num_validation :]

print('out of {0} samples for train, {1:0.2f}% are ones'.format(num_train, sum(final_train_targets) / num_train * 100))
print('out of {0} samples for validation, {1:0.2f}% are ones'.format(num_validation, sum(final_validation_targets) / num_validation * 100))
print('out of {0} samples for test, {1:0.2f}% are ones'.format(num_test, sum(final_test_targets) / num_test * 100))

np.savez('Audiobook_train_data', inputs=final_train_inputs, targets=final_train_targets)
np.savez('Audiobook_validation_data', inputs=final_validation_inputs, targets=final_validation_targets)
np.savez('Audiobook_test_data', inputs=final_test_inputs, targets=final_test_targets)