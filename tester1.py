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

before = cost_function(nn, t)
print(before)

learning_rate = 1.0
for _ in range(100):
    old_biases = nn.biases
    old_weights = nn.weights

    nab_b, nab_w = nn.backprop(x, y)
    nn.biases = [b - learning_rate * nb for b, nb in zip(nn.biases, nab_b)]
    nn.weights = [w - learning_rate * nw for w, nw in zip(nn.weights, nab_w)]

    after = cost_function(nn, t)

    if before < after:
        learning_rate *= 0.45
        nn.biases = old_biases
        nn.weights = old_weights
    else:
        learning_rate *= 1.05
        before = after

print(after)
print(nn.feedforward(x))
