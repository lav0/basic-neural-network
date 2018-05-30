from network import Network
from network import cost_function
from network import sigmoid
import numpy as np

nn = Network([2, 3, 1])
nn.biases = [np.array([[0], [0], [0]]), np.array([[0]])]
nn.weights = [
    np.array([[0, 0], [0, 0], [0, 0]]),
    np.array([[0, 0, 0]])
]

x = np.array([[0], [0]])

hi = nn.feedforward(x)
print(hi)