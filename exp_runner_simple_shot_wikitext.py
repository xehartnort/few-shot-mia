# Part1: Standard code to ensure reproducibility
import os

# os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
import numpy as np
import random
import gc

# Compute metrics
from sklearn.metrics import roc_curve
from tqdm import tqdm

from utils.model_tools import (
    FaissKNeighbors,
    compute_metrics,
    random_sample_wikited_dataset,
    load_data_NLP,
    load_train_model_NLP,
)

# Part2: Standard code to ensure reproducibility
SEED = 123
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


class Few_shot_model:
    def __init__(self, base_model, support_data, support_labels, avg_feat, k=1):
        self._base_model = base_model
        self._averaged_train_feat = avg_feat

        labels = np.argmax(support_labels, axis=1)

        tmp_features = self._base_model.predict(support_data, batch_size=2, verbose=False)
        tmp_features = np.array([i.flatten() for i in tmp_features])

        self._support_centroids = tmp_features
        self._support_labels = labels

        # Second big change, maybe we can add these as additional points?
        # compute centroids for each class
        if len(labels) > 2:  # big third
            # compute centroids for each class
            for i in np.unique(labels):
                selected_data = tmp_features[labels == i, :]
                selected_data = np.mean(selected_data, axis=0)
                self._support_centroids = np.append(
                    self._support_centroids, np.array([selected_data]), axis=0
                )
                self._support_labels = np.append(
                    self._support_labels, i
                )  # beware this label

        tmp_features = np.array([self._center_and_normalize(i) for i in tmp_features])

    def _center_and_normalize(self, data):
        n_data = data  # - self._averaged_train_feat
        n_data = n_data / np.maximum(np.linalg.norm(n_data), 1e-6)
        return n_data

    def __call__(self, query_data):

        feat = self._base_model.predict(query_data, batch_size=1, verbose=False)
        feat = np.array([self._center_and_normalize(i) for i in feat])
        return self._model.predict_proba(feat)

    def train(self, val_data, val_labels):
        # perform validation
        max_shots = len(self._support_centroids)

        feat = self._base_model.predict(val_data, batch_size=1, verbose=False)
        feat = np.array([self._center_and_normalize(f) for f in feat])

        d = len(query_data)
        half_d = d / 2
        best_score = 0
        sparse_labels = np.argmax(val_labels, axis=1)
        for k in range(1, max_shots + 1, 2):
            # tmp_model = KNeighborsClassifier(n_neighbors=k, weights='distance') # First change
            other_model = FaissKNeighbors(k=k)
            # tmp_model.fit(self._support_centroids, self._support_labels)
            other_model.fit(self._support_centroids, self._support_labels)
            # evaluate
            predictions = other_model.predict_proba(feat)
            # predictions = tmp_model.predict_proba(feat)
            probs = predictions[:, 1]

            fpr, tpr, _ = roc_curve(sparse_labels, probs)

            # Case A => our metric at FP = 0
            tps_at_very_low_fps = max(tpr[fpr <= 0]) * half_d

            if tps_at_very_low_fps >= best_score:
                best_score = tps_at_very_low_fps
                self._model = other_model


if __name__ == "__main__":

    query_samples = 15

    # Number of times to run the experiment
    experiment_times = 500

    # Run experiment collecting metrics
    all_metrics = []
    in_dataset, out_dataset = load_data_NLP()

    dataset = "wikitext"

    avg_feat = None  # np.mean(base_model.predict(in_dataset, batch_size=1, steps=len(in_dataset)), axis=0)
    for shots in [5, 10]:
        experiment_name = "experiments/simple_shot_{}.csv".format(dataset)

        # Write experiment header
        with open(experiment_name, "a+") as out:
            out.write("dataset, shots, query_samples, n_experiments\n")
            out.write(
                "{}, {}, {}, {}\n".format(
                    dataset, shots, query_samples, experiment_times
                )
            )
            out.write("auc, acc, precision, recall, case_b, case_a\n")

        # Load pretrained model with its training and non-training dataset
        trained_model = load_train_model_NLP()

        inputs = tf.keras.layers.Input((1, 1024,), dtype=tf.int32)
        logits = trained_model(inputs).logits
        x = tf.keras.layers.Flatten()(logits)
        base_model = tf.keras.Model(inputs=inputs, outputs=x)

        for i in tqdm(range(experiment_times)):

            # Take a random sample to create support and query dataset (training and test, respectively)
            (
                (support_data, support_labels),
                (val_data, val_labels),
                (query_data, query_labels),
            ) = random_sample_wikited_dataset(
                in_dataset, out_dataset, shots, query_samples
            )

            model = Few_shot_model(base_model, support_data, support_labels, avg_feat)

            model.train(val_data, val_labels)

            # Retrieve metrics
            auc, acc, precision, recall, case_b, case_a = compute_metrics(
                model, query_data, query_labels, val_data, val_labels
            )

            all_metrics.append([auc, acc, precision, recall, case_b, case_a])

        # write averaged results
        averaged_metrics = np.mean(all_metrics, axis=0)
        error = 1.96 * np.std(all_metrics, ddof=1, axis=0) / np.sqrt(experiment_times)
        with open(experiment_name, "a+") as out:
            out.write("--- Summary mean and 0.95 confidence interval---\n")
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*averaged_metrics)
            )
            out.write("{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*error))

        tf.keras.backend.clear_session()
        del base_model
        del model
        gc.collect()
