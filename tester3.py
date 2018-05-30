from network import Network
from network import cost_function
import numpy as np

nn = Network([3, 2, 2])
nn.biases = [np.array([[-1], [1]]), np.array([[1], [1]])]
nn.weights = [
    np.array([[1, -1, 1], [-1, 1, -1]]),
    np.array([[0.1, -0.5], [0.5, 0.6]])
]

x = np.array([[1], [2], [3]])
y = np.array([[0], [1]])

x1 = np.array([[0.9], [0], [1]])
y1 = np.array([[1], [0]])

t = x, y
t1 = x1, y1
t = [t, t1]

before = nn.feedforward(x)
print(before, cost_function(nn, t))

nn.SGD(training_data=t, epochs=1000, mini_batch_size=2, eta=1)

after = nn.feedforward(x)
after1 = nn.feedforward(x1)
print(after)
print(after1)
print(cost_function(nn, t))
