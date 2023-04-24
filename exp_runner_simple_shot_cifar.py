# Part1: Standard code to ensure reproducibility
import os

os.environ["TF_DETERMINISTIC_OPS"] = "1"

import tensorflow as tf
import numpy as np
import random
import argparse
from tqdm import tqdm

from utils.model_tools import *

# Part2: Standard code to ensure reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


class Few_shot_model:
    def __init__(self, base_model, support_data, support_labels, avg_feat):

        self._base_model = base_model
        self._averaged_train_feat = avg_feat

        labels = np.argmax(support_labels, axis=1)

        tmp_features = self._base_model.predict(support_data, verbose=False)

        self._support_centroids = tmp_features
        self._support_labels = labels

        # Second big change
        # compute centroids for each class
        if len(labels) > 2:  # big third
            for i in np.unique(labels):
                selected_data = tmp_features[labels == i, :]
                selected_data = np.mean(selected_data, axis=0)
                self._support_centroids = np.append(
                    self._support_centroids, np.array([selected_data]), axis=0
                )
                self._support_labels = np.append(
                    self._support_labels, i
                )  # beware this label is sparse

        tmp_features = np.array([self._center_and_normalize(t) for t in tmp_features])

    def _center_and_normalize(self, data):
        n_data = data - self._averaged_train_feat
        n_data = n_data / np.maximum(np.linalg.norm(n_data), 1e-6)
        return n_data

    def __call__(self, query_data):
        feat = self._base_model.predict(query_data, verbose=False)
        feat = np.array([self._center_and_normalize(f) for f in feat])
        return self._model.predict_proba(feat)

    def train(self, val_data, val_labels):
        # perform validation
        max_shots = int(len(self._support_centroids) / 2)

        feat = self._base_model.predict(val_data, verbose=False)
        feat = np.array([self._center_and_normalize(f) for f in feat])

        d = len(query_data)
        half_d = d / 2
        best_score = 0
        sparse_labels = np.argmax(val_labels, axis=1)
        for k in range(1, max_shots + 1, 2):
            tmp_model = FaissKNeighbors(k=k)
            # KNeighborsClassifier(n_neighbors=k, weights='distance') # First change
            tmp_model.fit(self._support_centroids, self._support_labels)

            # evaluate
            predictions = tmp_model.predict_proba(feat)
            probs = predictions[:, 1]

            fpr, tpr, _ = roc_curve(sparse_labels, probs)

            # Maximize case A
            # Case A => our metric at FP = 0
            tps_at_very_low_fps = max(tpr[fpr <= 0]) * half_d

            if tps_at_very_low_fps >= best_score:
                best_score = tps_at_very_low_fps
                self._model = tmp_model
                self._k = k

        return self._k


parser = argparse.ArgumentParser(description="TF Training")

parser.add_argument(
    "--dataset", type=int, default=0, help="use cifar100 [0], cifar10 [1]"
)

if __name__ == "__main__":

    args = parser.parse_args()
    print("Les argumentos", args)

    # Experiment epochs

    query_samples = 15  # How many query samples? this seem to be fixed

    # Tunable experiment configuration
    # Setup and configuration for few-shot experiment
    cifar10 = bool(args.dataset)  # otherwise, use cifar100

    oversample = 0
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
    avg_feat = np.mean(base_model.predict(in_data, verbose=False), axis=0)

    for shots in [5, 10]:
        dataset = "cifar10" if cifar10 else "cifar100"
        experiment_name = "experiments/simple_shot_WRN{}_{}_{}.csv".format(
            k, n, dataset
        )

        # Run experiment collecting metrics
        all_metrics = []

        # Write experiment header
        with open(experiment_name, "a+") as out:
            out.write("dataset, shots, query_samples, n_experiments, \n")
            out.write(
                "{}, {}, {}, {}\n".format(
                    dataset, shots, query_samples, experiment_times
                )
            )
            out.write("k, auc, acc, precision, recall, case_b, case_a\n")

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
            k_ = model.train(val_data, val_labels)

            # Retrieve metrics, choosing the threshold in validation
            auc, acc, precision, recall, case_b, case_a = compute_metrics(
                model, query_data, query_labels, val_data, val_labels
            )

            all_metrics.append([k_, auc, acc, precision, recall, case_b, case_a])

        # write averaged results
        averaged_metrics = np.mean(all_metrics, axis=0)
        error = 1.96 * np.std(all_metrics, ddof=1, axis=0) / np.sqrt(experiment_times)
        with open(experiment_name, "a+") as out:
            out.write("--- Summary mean and 0.95 confidence interval---\n")
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    *averaged_metrics
                )
            )
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*error)
            )
