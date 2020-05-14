import random
import time
import numpy as np
import pandas as pd

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler


class Iteration:
    def __init__(self, filepath, size, active_clients, alpha, epsilon, mean, dp_algorithm=None, intercept_dp=False):
        self.df = pd.read_csv(filepath)
        self.features = ['age', 'bmi', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']
        self.label = ['t2d']
        self.size = size
        self.completed_round = 0
        self.dp_algorithm = dp_algorithm
        self.intercept_dp = intercept_dp
        self.active_clients = active_clients
        self.alpha = alpha
        self.epsilon = epsilon
        self.mean = mean

    def begin_round(self, random_state, federated_weights=None, federated_intercepts=None):
        start_time = time.time()

        weights, intercepts = self.compute_weights(random_state, federated_weights, federated_intercepts)

        if self.dp_algorithm:
            weights, intercepts = self.add_noise(random_state=random_state, weights=weights, intercepts=intercepts)

        end_time = time.time()
        computation_time = end_time - start_time
        # TODO: Add this to DB
        self.completed_round += 1

        return {'weights': weights.tolist(), 'intercepts': intercepts.tolist(), 'iteration': self.completed_round,
                'computation_time': computation_time * 1000}

    def compute_weights(self, random_state, federated_weights, federated_intercepts):
        sample_begin = self.completed_round * self.size
        sample_end = sample_begin + self.size

        if sample_end >= len(self.df):
            raise ValueError('Not enough samples available for training round')

        X = pd.DataFrame(self.df, columns=self.features)[sample_begin:sample_end]
        y = pd.DataFrame(self.df, columns=self.label)[sample_begin:sample_end]
        y = y.values.ravel()

        scaler = MinMaxScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        lr = SGDClassifier(alpha=0.0001, loss="log", random_state=random_state)

        lr.fit(X, y, coef_init=federated_weights, intercept_init=federated_intercepts)
        local_weights = lr.coef_
        local_intercepts = lr.intercept_

        return local_weights, local_intercepts

    def add_noise(self, random_state, weights, intercepts):
        weights_shape = weights.shape
        weights_dp_noise = np.zeros(weights_shape)

        intercepts_shape = intercepts.shape
        intercepts_dp_noise = np.zeros(intercepts_shape)

        sensitivity = 2 / (self.active_clients * self.size * self.alpha)

        random.seed(random_state)
        for i in range(weights_shape[0]):
            for j in range(weights_shape[1]):
                if self.dp_algorithm == 'Laplace':
                    dp_noise = self.laplace(sensitivity)
                elif self.dp_algorithm == 'Gamma':
                    scale = sensitivity / self.epsilon
                    dp_noise = random.gammavariate(1 / self.active_clients, scale) - random.gammavariate(
                        1 / self.active_clients,
                        scale)
                else:
                    raise AssertionError('Invalid differential privacy algorithm in config, use Laplace or Gamma')
                weights_dp_noise[i][j] = dp_noise

        if self.intercept_dp:
            for i in range(intercepts_shape[0]):
                if self.dp_algorithm == 'Laplace':
                    dp_noise = self.laplace(sensitivity)
                elif self.dp_algorithm == 'Gamma':
                    scale = sensitivity / self.epsilon
                    dp_noise = random.gammavariate(1 / self.active_clients, scale) - random.gammavariate(
                        1 / self.active_clients, scale)
                else:
                    raise AssertionError('Invalid differential privacy algorithm in config, use Laplace or Gamma')
                intercepts_dp_noise[i] = dp_noise

        weights += weights_dp_noise
        intercepts += intercepts_dp_noise
        return weights, intercepts

    def laplace(self, sensitivity):
        scale = sensitivity / self.epsilon
        rand = random.uniform(0, 1) - 0.5
        return self.mean - scale * np.sign(rand) * np.log(1 - 2 * np.abs(rand))
