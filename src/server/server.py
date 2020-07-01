import requests
import json
import numpy as np
import random
from evaluator import ModelEvaluation

BASE_URL = 'http://0.0.0.0:'
ITERATIONS = 7
STUDY_NAME = "MyStudy"
SAMPLE_SIZE = 100
ENABLE_DP = False
TOLERANCE = 20.0
# Client to port mappings, this can be a dynamic registration list based on IP Addresses too
CLIENTS = {1: 5001, 2: 5002, 3: 5003}
TESTFILE = '1kg_pca_pheno_test.csv'


class Study:
    def __init__(self):
        self.active_clients = {}
        self.federated_weights = {0: np.empty((0, 0))}
        self.federated_intercepts = {0: np.empty((0, 0))}
        self.client_weights = {}
        self.client_intercepts = {}
        self.client_computation_times = {}

    def registration(self):
        for client in CLIENTS.keys():
            # register study with each client
            # if failed to register then remove client from consideration
            body = {
                "name": STUDY_NAME,
                "dp": ENABLE_DP,
                "size": SAMPLE_SIZE
            }
            try:
                resp = requests.post(BASE_URL + str(CLIENTS[client]) + '/register', json=body)
                if resp.status_code == 200:
                    self.active_clients[client] = CLIENTS[client]
                else:
                    # log reason
                    pass
            except Exception as e:
                # log reason
                pass

    def federation(self):

        active_clients = CLIENTS
        random.seed(0)
        # RANDOM_SEEDS: required for reproducibility of simulation. Seeds every iteration of the training for each client
        random_seeds = {client: list(random.sample(range(0, 1000000), 100)) for client in active_clients}

        print("Begin Federated Learning Rounds\n")
        for i in range(ITERATIONS):
            for client in active_clients.keys():
                # do iteration per client and average out their weights
                body = {
                    "name": STUDY_NAME,
                    "weights": self.federated_weights[i].tolist(),
                    "intercepts": self.federated_intercepts[i].tolist(),
                    "clients": len(active_clients),
                    "seed": random_seeds[client][i]
                }
                # TODO: Add timeout for response
                resp = requests.post(BASE_URL + str(active_clients[client]) + '/train', json=body)

                if resp.status_code == 200:
                    result = json.loads(resp.text)
                    self.client_weights[client] = np.asarray(result['weights'])
                    self.client_intercepts[client] = np.asarray(result['intercepts'])
                    self.client_computation_times[client] = result["computation_time"]
                else:
                    # log reason
                    continue

            # Perform federation and average across all clients
            weights_np = list(self.client_weights.values())  # the weights for this iteration!
            intercepts_np = list(self.client_intercepts.values())
            try:
                averaged_weights = np.average(weights_np, axis=0)  # gets rid of security offsets
            except:
                raise ValueError('''DATA INSUFFICIENT: Some client does not have a sample from each class so dimension of weights is incorrect. Make
                                             train length per iteration larger for each client to avoid this issue''')

            averaged_intercepts = np.average(intercepts_np, axis=0)
            self.federated_weights[i + 1] = averaged_weights
            self.federated_intercepts[i + 1] = averaged_intercepts
            self.print_iteration(i)
            # TODO: add timeouts

    def print_iteration(self, i):
        md = ModelEvaluation(TESTFILE)
        print("Iteration " + str(i + 1) + "\n")
        for c in self.client_weights.keys():
            print("--------------------------------------------------------------")
            print("Client " + str(c))
            print("--------------------------------------------------------------")
            print("Local Accuracy: " + str(md.test(self.client_weights[c], self.client_intercepts[c])))
            print("Round Trip Time (ms): " + str(self.client_computation_times[c]))
            convergence = md.converged((self.client_weights[c], self.client_intercepts[c]),
                                       (self.federated_weights[i + 1], self.federated_intercepts[i + 1]),
                                       TOLERANCE)

        print("###############################################################")
        print("Federated Model Accuracy: " + str(
            md.test(self.federated_weights[i + 1], self.federated_intercepts[i + 1])))
        print("###############################################################")
        print("\n")

obj = Study()
obj.registration()
obj.federation()
