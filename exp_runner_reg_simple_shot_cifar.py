# Part1: Standard code to ensure reproducibility
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
import numpy as np
import random
from scipy import sparse
import argparse
from tqdm import tqdm

from utils.model_tools import *

# Part2: Standard code to ensure reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


def normalize(Y_in):
    maxcol = np.max(Y_in, axis=1)
    Y_in = Y_in - maxcol[:, np.newaxis]
    Y_out = np.exp(Y_in)
    Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

    return Y_out


def entropy_energy(Y, unary, kernel, bound_lambda, batch=False):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)
    temp = (unary * Y) + (-bound_lambda * pairwise * Y)
    E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()

    return E


def bound_update(unary, kernel_affinity, bound_lambda, bound_iteration=20, batch=False):
    oldE = float("inf")
    Y = normalize(-unary)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel_affinity.dot(Y)
        Y = -bound_lambda * mul_kernel
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy(Y, unary, kernel_affinity, bound_lambda, batch)
        E_list.append(E)
        # print('entropy_energy is ' +repr(E) + ' at iteration ',i)
        if i > 1 and (abs(E - oldE) <= 1e-6 * abs(oldE)):
            # print('Converged')
            break

        else:
            oldE = E.copy()
    # plot_convergence(E_list)
    return Y


class Few_shot_model:
    def __init__(self, base_model, support_data, support_labels, avg_feat=None):
        self._base_model = base_model
        self._averaged_train_feat = avg_feat
        self._norm = avg_feat is not None
        self._k = 3
        self._lambda = 1

        tmp_features = self._base_model.predict(support_data, verbose=False)

        labels = np.argmax(support_labels, axis=1)
        self._support_centroids = list(tmp_features)
        self._support_labels = list(labels)

        # compute centroids for each class
        if len(labels) > 2:  # big third
            for i in np.unique(labels):
                selected_data = tmp_features[labels == i, :]
                selected_data = np.mean(selected_data, axis=0)
                self._support_centroids.append(selected_data)
                self._support_labels.append(i)  # beware this label is sparse

        self._support_centroids = np.array(self._support_centroids)
        self._support_labels = np.array(self._support_labels)

        self.__nbrs = FaissKNeighbors(k=len(self._support_centroids)).fit(
            self._support_centroids
        )

    def _create_affinity(self, X, knn):  # did i change k here?: Nothing
        N, _ = X.shape

        nbrs = FaissKNeighbors(k=knn).fit(X)
        _, knnind = nbrs.kneighbors(X)

        row = np.repeat(range(N), knn - 1)
        col = knnind[:, 1:].flatten()
        data = np.ones(X.shape[0] * (knn - 1))

        # W[row[k], col[k]] = data[k]
        W = sparse.csc_matrix((data, (row, col)), shape=(N, N), dtype=float)
        return W

    def _lshot_prediction(self, knn, lmd, X, unary):

        self.W = self._create_affinity(X, knn)
        Y = bound_update(unary, self.W, lmd)
        # l = np.argmax(Y, axis=1)

        # labels = np.take(self._support_labels, l)

        ones_prob = np.sum(Y[:, self._support_labels == 1], axis=1)
        zeros_prob = np.sum(Y[:, self._support_labels == 0], axis=1)

        return np.transpose([zeros_prob, ones_prob])

    def __call__(self, query_data):
        query_features = self._base_model.predict(query_data, verbose=False)
        # distancias de los centroides a cada uno de los elementos
        # subtract = self._support_centroids[:,None,:] - query_features
        # distance = np.linalg.norm(subtract, 2, axis=-1)
        # unary = distance.transpose() ** 2

        unary = self.__nbrs.distance_matrix(
            query_features
        )  # Cada fila, tiene la distancias de todos los centroides a una feature

        pred = self._lshot_prediction(self._k, self._lambda, query_features, unary)

        return pred

    def train(self, val_data, val_labels):
        # Choose best lambda value
        lmd_list = [1.5, 1.7, 2, 2.2, 2.5, 2.7, 2.9]
        k_values = [3, 5, 7, 9, 11, 13, 15]

        d = len(query_data)
        max_FP = round(np.log(d), 0)
        half_d = d / 2
        best_score = 0
        sparse_labels = np.argmax(val_labels, axis=1)

        feat = self._base_model.predict(val_data, verbose=False)
        if self._norm:
            feat = self._center_and_normalize(feat)

        unary = self.__nbrs.distance_matrix(feat)

        for k in k_values:
            for l in lmd_list:
                # evaluate
                predictions = self._lshot_prediction(k, l, feat, unary)
                probs = predictions[:, 1]

                fpr, tpr, _ = roc_curve(sparse_labels, probs)

                # Maximize case A
                # Case A => our metric at FP = 0
                tps_at_very_low_fps = max(tpr[fpr <= 0]) * half_d

                if tps_at_very_low_fps >= best_score:
                    best_score = tps_at_very_low_fps
                    self._lambda = l
                    self._k = k

        return self._lambda, self._k


parser = argparse.ArgumentParser(description="TF Training")

parser.add_argument(
    "--dataset", type=int, default=0, help="use cifar100 [0], cifar10 [1]"
)

parser.add_argument(
    "--norm", type=int, default=0, help="use 0 for disable normalization"
)

if __name__ == "__main__":

    args = parser.parse_args()
    print("Les argumentos", args)

    # Experiment epochs
    query_samples = 15  # How many query samples? this seem to be fixed

    # Tunable experiment configuration
    # Setup and configuration for few-shot experiment
    norm = bool(args.norm)  # wether the samle
    cifar10 = bool(args.dataset)  # otherwise, use cifar100

    # Number of times to run the experiment
    experiment_times = 500

    # WRN config
    k = 28
    n = 2

    # Load pretrained model with its training and non-training dataset
    base_model, (in_data, in_labels), (out_data, out_labels) = load_train_model(
        cifar10, k=k, n=n
    )
    # Take a random sample to create support and query dataset (training and test, respectively)
    feature = base_model.layers[-2].output
    base_model = tf.keras.Model(inputs=base_model.input, outputs=feature)

    if norm:
        avg_feat = np.mean(
            base_model.predict(in_data, batch_size=256, verbose=False), axis=0
        )
    else:
        avg_feat = None
    for shots in [1, 5, 10]:

        dataset = "cifar10" if cifar10 else "cifar100"

        experiment_name = "experiments/reg_simple_shot_WRN{}_{}_{}.csv".format(
            k, n, dataset
        )

        # Run experiment collecting metrics
        all_metrics = []

        # Write experiment header
        with open(experiment_name, "a+") as out:
            out.write("dataset, shots, query_samples, n_experiments, norm\n")
            out.write(
                "{}, {}, {}, {}, {}\n".format(
                    dataset, shots, query_samples, experiment_times, norm
                )
            )
            out.write("lambda, knn, auc, acc, precision, recall, case_b, case_a\n")

        for i in tqdm(range(experiment_times)):

            (
                (support_data, support_labels),
                (val_data, val_labels),
                (query_data, query_labels),
            ) = random_sample_datasets(in_data, out_data, shots, query_samples)

            model = Few_shot_model(
                base_model, support_data, support_labels, avg_feat
            )  # third big change

            # Find best K
            _l, _k = model.train(val_data, val_labels)

            # Retrieve metrics, choosing the threshold in validation
            auc, acc, precision, recall, case_b, case_a = compute_metrics(
                model, query_data, query_labels, val_data, val_labels
            )

            all_metrics.append([_l, _k, auc, acc, precision, recall, case_b, case_a])

        # write averaged results
        averaged_metrics = np.mean(all_metrics, axis=0)
        error = 1.96 * np.std(all_metrics, ddof=1, axis=0) / np.sqrt(experiment_times)
        with open(experiment_name, "a+") as out:
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    *averaged_metrics
                )
            )
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    *error
                )
            )
