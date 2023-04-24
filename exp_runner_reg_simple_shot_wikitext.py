# Part1: Standard code to ensure reproducibility
import os

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
import numpy as np
import random
import gc
import math

from tqdm import tqdm
from transformers import TFGPT2LMHeadModel as gpt2, GPT2Config
from datasets import load_from_disk
from transformers import DefaultDataCollator
from scipy import sparse
from sklearn.metrics import roc_curve
import argparse

from utils.model_tools import (
    FaissKNeighbors,
    compute_metrics,
    random_sample_wikited_dataset,
    load_data_NLP,
    load_train_model_NLP,
)

# Part2: Standard code to ensure reproducibility
SEED = 256
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


def normalize(Y_in):
    maxcol = np.max(Y_in, axis=1)
    Y_in = Y_in - maxcol[:, np.newaxis]
    N = Y_in.shape[0]
    size_limit = 150000
    if N > size_limit:
        batch_size = 1280
        Y_out = []
        num_batch = int(math.ceil(1.0 * N / batch_size))
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, N)
            tmp = np.exp(Y_in[start:end, :])
            tmp = tmp / (np.sum(tmp, axis=1)[:, None])
            Y_out.append(tmp)
        del Y_in
        Y_out = np.vstack(Y_out)
    else:
        Y_out = np.exp(Y_in)
        Y_out = Y_out / (np.sum(Y_out, axis=1)[:, None])

    return Y_out


def entropy_energy(Y, unary, kernel, bound_lambda, batch=False):
    tot_size = Y.shape[0]
    pairwise = kernel.dot(Y)
    if batch == False:
        temp = (unary * Y) + (-bound_lambda * pairwise * Y)
        E = (Y * np.log(np.maximum(Y, 1e-20)) + temp).sum()
    else:
        batch_size = 1024
        num_batch = int(math.ceil(1.0 * tot_size / batch_size))
        E = 0
        for batch_idx in range(num_batch):
            start = batch_idx * batch_size
            end = min((batch_idx + 1) * batch_size, tot_size)
            temp = (unary[start:end] * Y[start:end]) + (
                -bound_lambda * pairwise[start:end] * Y[start:end]
            )
            E = (
                E
                + (Y[start:end] * np.log(np.maximum(Y[start:end], 1e-20)) + temp).sum()
            )

    return E


def bound_update(unary, kernel, bound_lambda, bound_iteration=20, batch=False):
    """
    """
    oldE = float("inf")
    Y = normalize(-unary)
    E_list = []
    for i in range(bound_iteration):
        additive = -unary
        mul_kernel = kernel.dot(Y)
        Y = -bound_lambda * mul_kernel
        additive = additive - Y
        Y = normalize(additive)
        E = entropy_energy(Y, unary, kernel, bound_lambda, batch)
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
    def __init__(self, base_model, support_data, support_labels, norm=False):
        self._base_model = base_model
        self._norm = norm
        self._k = 3
        self._lambda = 1

        tmp_features = self._base_model.predict(support_data, batch_size=5, verbose=False)

        labels = np.argmax(support_labels, axis=1)
        self._support_centroids = list(tmp_features)
        self._support_labels = list(labels)

        # compute centroids for each class
        if len(labels) > 2:
            for i in np.unique(labels):
                selected_data = tmp_features[labels == i, :]
                selected_data = np.mean(selected_data, axis=0)
                self._support_centroids.append(selected_data)
                self._support_labels.append(i)  # beware this label is sparse

        self._support_centroids = np.array(self._support_centroids)
        self._support_labels = np.array(self._support_labels)
        if norm:
            self._support_centroids = np.array(
                [self._center_and_normalize(t) for t in self._support_centroids]
            )
        self.__nbrs = FaissKNeighbors(k=len(self._support_centroids)).fit(
            self._support_centroids
        )

    def _center_and_normalize(self, data):
        # n_data = data - self._averaged_train_feat
        n_data = data / np.maximum(np.linalg.norm(data), 1e-6)
        return n_data

    def __call__(self, query_data):
        query_features = self._base_model.predict(query_data, batch_size=5)
        # query_features=np.array([self._center_and_normalize(t) for t in query_features])

        # query_features = np.array([i.flatten() for i in query_features])

        # subtract = self._support_centroids[:,None,:] - query_features
        # distance = np.linalg.norm(subtract, 2, axis=-1)
        # unary = distance.transpose() ** 2

        # distancias de los centroides a cada uno de los elementos
        unary = self.__nbrs.distance_matrix(
            query_features
        )  # Cada fila, tiene la distancias de todos los centroides a una feature

        pred = self._lshot_prediction(self._k, self._lambda, query_features, unary)

        return pred

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

        W = self._create_affinity(X, knn)
        Y = bound_update(unary, W, lmd)
        # l = np.argmax(Y, axis=1)

        # labels = np.take(self._support_labels, l)

        ones_prob = np.sum(Y[:, self._support_labels == 1], axis=1)
        zeros_prob = np.sum(Y[:, self._support_labels == 0], axis=1)

        return np.transpose([zeros_prob, ones_prob])

    def train(self, val_data, val_labels):

        # Choose best lambda value
        lmd_list = [1, 1.5, 2.5, 2.9]
        k_values = [3, 5, 9, 11, 13]

        d = len(query_data)
        half_d = d / 2
        best_score = 0
        val_sparse_labels = np.argmax(val_labels, axis=1)

        feat = self._base_model.predict(val_data, batch_size=5, verbose=False)

        unary = self.__nbrs.distance_matrix(
            feat
        )  # Cada fila, tiene la distancias de todos los centroides a una feature

        # subtract = self._support_centroids[:,None,:] - feat
        # distance = np.linalg.norm(subtract, 2, axis=-1)
        # unary = distance.transpose() ** 2

        # for l, k in tqdm(itertools.product(lmd_list, k_values)):
        for k in k_values:
            for l in lmd_list:
                # evaluate
                predictions = self._lshot_prediction(
                    k, l, feat, unary
                )  # I have one issue, probabilities are discrete
                probs = predictions[:, 1]

                fpr, tpr, _ = roc_curve(val_sparse_labels, probs)

                # Case A => our metric at FP = 0
                tps_at_very_low_fps = max(tpr[fpr <= 0]) * half_d
                print("k, l, low", k, l, tps_at_very_low_fps)
                if tps_at_very_low_fps >= best_score:
                    best_score = tps_at_very_low_fps
                    self._lambda = l
                    self._k = k

        return self._lambda, self._k


parser = argparse.ArgumentParser(description="TF Training")

parser.add_argument(
    "--shots", type=int, default=1, help="use 0 for disable normalization"
)


if __name__ == "__main__":

    args = parser.parse_args()
    print("Les argumentos", args)

    query_samples = 15

    # Number of times to run the experiment
    experiment_times = (
        500  # preliminar estudy of which k-values and lambda values are required
    )

    # Run experiment collecting metrics
    all_metrics = []
    in_dataset, out_dataset = load_data_NLP()

    all_shots = args.shots

    # Setup and configuration for few-shot experiment
    norm = False  # wether the samle
    dataset = "wikitext"

    # Load pretrained model with its training and non-training dataset
    trained_model = load_train_model_NLP()

    inputs = tf.keras.layers.Input((1, 1024,), dtype=tf.int32)
    logits = trained_model(inputs).logits
    x = tf.keras.layers.Flatten()(logits)
    base_model = tf.keras.Model(inputs=inputs, outputs=x)

    for shots in [all_shots]:
        experiment_name = "experiments/reg_simple_shot_{}.csv".format(dataset)

        # Write experiment header
        with open(experiment_name, "a+") as out:
            out.write("dataset, shots, query_samples, n_experiments, norm\n")
            out.write(
                "{}, {}, {}, {}, {}\n".format(
                    dataset, shots, query_samples, experiment_times, norm
                )
            )
            out.write("lambda, knn, auc, acc, precision, recall, case_b, case_a\n")

        pbar = tqdm(range(experiment_times))
        for i in pbar:

            # Take a random sample to create support and query dataset (training and test, respectively)
            (
                (support_data, support_labels),
                (val_data, val_labels),
                (query_data, query_labels),
            ) = random_sample_wikited_dataset(
                in_dataset, out_dataset, shots, query_samples
            )

            model = Few_shot_model(base_model, support_data, support_labels, norm=norm)

            # Find best K
            _l, _k = model.train(val_data, val_labels)

            # Retrieve metrics, choosing the threshold in validation
            auc, acc, precision, recall, case_b, case_a = compute_metrics(
                model, query_data, query_labels, val_data, val_labels
            )

            all_metrics.append([_l, _k, auc, acc, precision, recall, case_b, case_a])

            tmp_avg = np.mean(all_metrics, axis=0)
            pbar.set_postfix(
                {
                    "lambda": tmp_avg[0],
                    "avg_case_b": tmp_avg[-2],
                    "avg_case_a": tmp_avg[-1],
                }
            )

            tf.keras.backend.clear_session()
            del model
            gc.collect()

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
