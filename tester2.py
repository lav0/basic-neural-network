import numpy as np
from network import Network

data = np.loadtxt("data.csv", delimiter=",")

test_index = np.random.choice([True, False], len(data), replace=True, p=[0.25, 0.75])
test = data[test_index]
train = data[10: 40]
train = [(d[:3][:, np.newaxis], np.eye(3, 1, k=-int(d[-1]))) for d in train]
test = [(d[:3][:, np.newaxis], d[-1]) for d in test]
input_count = 3  # 3 нейрона входного слоя
hidden_count = 6  # 5 нейронов внутреннего слоя
output_count = 3
nn = Network([input_count, hidden_count, output_count])
print(train[0])
nn.SGD(training_data=train, epochs=1, mini_batch_size=1, eta=1)
