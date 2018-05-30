import random
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    sig_tmp = sigmoid(z)
    return sig_tmp * (1 - sig_tmp)


def cost_function(network, test_data, onehot=True):
    c = 0
    for example, y in test_data:
        if not onehot:
            y = np.eye(3, 1, k=-int(y))
        yhat = network.feedforward(example)
        c += np.sum((y - yhat)**2)
    return c / len(test_data)


class Network:
    def __init__(self, sizes, output_funtion=sigmoid, output_derivative=sigmoid_prime):
        """
        Список ``sizes`` содержит количество нейронов в соответствующих слоях
        нейронной сети. К примеру, если бы этот лист выглядел как [2, 3, 1],
        то мы бы получили трёхслойную нейросеть, с двумя нейронами в первом
        (входном), тремя нейронами во втором (промежуточном) и одним нейроном
        в третьем (выходном, внешнем) слое. Смещения и веса для нейронных сетей
        инициализируются случайными значениями, подчиняющимися стандартному нормальному
        распределению. Обратите внимание, что первый слой подразумевается слоем,
        принимающим входные данные, поэтому мы не будем добавлять к нему смещение
        (делать это не принято, поскольку смещения используются только при
        вычислении выходных значений нейронов последующих слоёв)
        """

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
        self.output_function = output_funtion
        self.output_derivative = output_derivative

    def feedforward(self, a):
        """
        Вычислить и вернуть выходную активацию нейронной сети
        при получении ``a`` на входе (бывшее forward_pass).
        """
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)

        output = np.dot(self.weights[-1], a)
        output += self.biases[-1]

        return self.output_function(output)

    def SGD(self, training_data, epochs, mini_batch_size, eta):
        """
        Обучить нейронную сеть, используя алгоритм стохастического
        (mini-batch) градиентного спуска.
        ``training_data`` - лист кортежей вида ``(x, y)``, где
        x - вход обучающего примера, y - желаемый выход (в формате one-hot).
        Роль остальных обязательных параметров должна быть понятна из их названия.
        Если предоставлен опциональный аргумент ``test_data``,
        то после каждой эпохи обучения сеть будет протестирована на этих данных
        и промежуточный результат обучения будет выведен в консоль.
        ``test_data`` -- это список кортежей из входных данных
        и номеров правильных классов примеров (т.е. argmax(y),
        если y -- набор ответов в той же форме, что и в тренировочных данных).
        Тестирование полезно для мониторинга процесса обучения,
        но может существенно замедлить работу программы.
        """

        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

    def update_mini_batch(self, mini_batch, eta):
        """
        Обновить веса и смещения нейронной сети, сделав шаг градиентного
        спуска на основе алгоритма обратного распространения ошибки, примененного
        к одному mini batch.
        ``mini_batch`` - список кортежей вида ``(x, y)``,
        ``eta`` - величина шага (learning rate).
        """

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        eps = eta / len(mini_batch)
        self.weights = [w - eps * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - eps * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """
        Возвращает кортеж ``(nabla_b, nabla_w)`` -- градиент целевой функции по всем параметрам сети.
        ``nabla_b`` и ``nabla_w`` -- послойные списки массивов ndarray,
        такие же, как self.biases и self.weights соответственно.
        """
        # Эту функцию необходимо реализовать
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # прямое распространение (forward pass)
        activation = x
        activations = [x]
        zs = []

        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            z = w.dot(activation) + b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        z = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z)
        output = self.output_function(z)
        activations.append(output)

        # обратное распространение (backward pass)
        delta = self.cost_derivative(activations[-1], y) * self.output_derivative(zs[-1])  # ошибка выходного слоя
        nabla_b[-1] = delta  # производная J по смещениям выходного слоя
        nabla_w[-1] = delta.dot(activations[-2].T)  # производная J по весам выходного слоя

        # Обратите внимание, что переменная l в цикле ниже используется
        # немного иначе, чем в лекциях.  Здесь l = 1 означает последний слой,
        # l = 2 - предпоследний и так далее.
        # Мы перенумеровали схему, чтобы с удобством для себя
        # использовать тот факт, что в Python к переменной типа list
        # можно обращаться по негативному индексу.
        for l in range(2, self.num_layers):
            # дополнительные вычисления, чтобы легче записывалось
            #
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp  # ошибка на слое L-l
            nabla_b[-l] = delta  # производная J по смещениям L-l-го слоя
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)  # производная J по весам L-l-го слоя
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        """
        Вернуть количество тестовых примеров, для которых нейронная сеть
        возвращает правильный ответ. Обратите внимание: подразумевается,
        что выход нейронной сети - это индекс, указывающий, какой из нейронов
        последнего слоя имеет наибольшую активацию.
        """
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """
        Возвращает вектор частных производных (\partial C_x) / (\partial a)
        целевой функции по активациям выходного слоя.
        """
        return (output_activations - y)
