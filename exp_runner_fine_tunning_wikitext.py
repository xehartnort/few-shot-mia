# Part1: Standard code to ensure reproducibility
import os

# os.environ["CUDA_VISIBLE_DEVICES"]='0'
# os.environ['TF_DETERMINISTIC_OPS'] = '1'

import tensorflow as tf
import numpy as np
import random
import gc
from tqdm import tqdm
from sklearn.metrics import roc_curve
import argparse
from utils.model_tools import (
    compute_metrics,
    SupportInitNLP,
    random_sample_wikited_dataset,
    load_data_NLP,
    load_train_model_NLP,
)

# Part2: Standard code to ensure reproducibility
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.run_functions_eagerly(True)
# tf.keras.mixed_precision.set_global_policy('mixed_float16')

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
    def __init__(self, base_model, support_data, support_labels, temp=0.8):

        inputs = tf.keras.layers.Input((1, 1024,), dtype=tf.int32)
        logits = base_model(inputs).logits
        # logits = tf.keras.layers.Flatten()(logits)
        def hola(x):
            return tf.squeeze(x[:,:,-1,:], axis=1)
        logits = tf.keras.layers.Lambda(hola)(logits)
        logits = tf.keras.layers.ReLU()(logits)
        logits = tf.keras.layers.Dense(
            units=2,  # number of classes: in and out
            kernel_initializer=SupportInitNLP(base_model, support_data, support_labels),
            bias_initializer="zeros",
            activation=None,
            name="dense_output",
        )(logits)

        self._model = tf.keras.Model(inputs=inputs, outputs=logits)
        self._temp = temp

        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)

        self._model.compile(optimizer, loss_fn, metrics=["accuracy"])
        self._support_data = support_data
        self._support_labels = support_labels

        self._support_dataset = tf.data.Dataset.from_tensor_slices(
            (support_data, support_labels)
        )

        def shannon_entropy(p, temp=self._temp):
            eps = 1e-10
            p = tf.nn.softmax(p / temp)
            p_ = tf.math.maximum(p, eps)
            a = -tf.math.reduce_sum(p * tf.math.log(p_), axis=1)
            return tf.math.reduce_mean(a)

        self._shannon_entropy = shannon_entropy

    def __call__(self, query_data):
        s = self._model.predict(query_data, batch_size=30, verbose=False)
        return tf.nn.softmax(s)

    def _fast_train(self, query_data, val_data, val_labels):
        batch_size = min(len(self._support_data), 5)
        tmp_ds = (
            self._support_dataset.repeat(1)
            .shuffle(buffer_size=batch_size)
            .batch(batch_size)
        )

        real_loss = []
        for x, y in tmp_ds:
            with tf.GradientTape() as tape:
                logits = self._model(x, training=True)  # Logits for this minibatch
                loss_value = self._model.loss(y, logits)

            grads = tape.gradient(loss_value, self._model.trainable_weights)
            self._model.optimizer.apply_gradients(
                zip(grads, self._model.trainable_weights)
            )

            real_loss.append(loss_value.numpy())

        real_shannon_loss = []
        for i in range(6): # batch size of 5
            with tf.GradientTape() as tape:
                soft_logits = self._model(query_data[5*i:(i+1)*5], training=True)
                shannon_loss = self._shannon_entropy(soft_logits, temp=self._temp)

            grads = tape.gradient(shannon_loss, self._model.trainable_weights)
            self._model.optimizer.apply_gradients(
                zip(grads, self._model.trainable_weights)
            )
            real_shannon_loss.append(shannon_loss)

        val_loss, _ = self._model.evaluate(
            val_data, val_labels, batch_size=10, verbose=False
        )

        return np.mean(real_loss), np.mean(real_shannon_loss), val_loss

    def train(
        self, epochs, query_data, val_data, val_labels, verbose=False
    ):
        patience = 3

        # Test a few temp values and choose the best one
        temps = [3, 5, 7, 9]

        bk_model_path = "{}.h5".format(id(self))
        best_model_path = "{}_best.h5".format(id(self))
        self._model.save_weights(bk_model_path)

        val_sparse_labels = np.argmax(val_labels, axis=1)
        d = 30
        half_d = d / 2

        best_score = 0
        best_score_2 = 0
        for t in temps:
            self._temp = t
            # Restore model state to a checkpoint
            self._model.load_weights(bk_model_path)

            # Train it with a new temp configuration
            wait = 0
            best = float("inf")
            for _ in range(epochs):
                _, _, val_loss = self._fast_train(
                    query_data, val_data, val_labels
                )
                wait += 1
                if val_loss < best:
                    best = val_loss
                    wait = 0
                if wait >= patience:
                    break

            # evaluate current model
            val_predictions = self(val_data)
            val_probs = val_predictions[:, 1]
            fpr, tpr, _ = roc_curve(val_sparse_labels, val_probs)

            # Case A => our metric at FP = 0
            tps_at_very_low_fps = max(tpr[fpr <= 0]) * half_d
            tps = tpr*half_d
            fps = fpr*half_d
            tps_at_low_fps = max(tps[fps <= np.ceil(np.log(15))])

            if tps_at_very_low_fps > best_score:
                best_score = tps_at_very_low_fps
                self._model.save_weights(best_model_path)
                best_temp = t
            elif tps_at_very_low_fps == best_score:
                if tps_at_low_fps >= best_score_2:
                    best_score_2 = tps_at_low_fps
                    self._model.save_weights(best_model_path)
                    best_temp = t

            print(f"Result A{tps_at_very_low_fps}, B {tps_at_low_fps}, temp:{t}")

        # Load best model weights, save best temperature value
        self._model.load_weights(best_model_path)
        self._temp = best_temp

        if os.path.exists(bk_model_path):
            os.remove(bk_model_path)

        if os.path.exists(best_model_path):
            os.remove(best_model_path)

        return self._temp


parser = argparse.ArgumentParser(description="TF Training")

parser.add_argument(
    "--shots", type=int, default=1, help="number of shots used for training"
)
parser.add_argument(
    "--epochs", type=int, default=30, help="number of epochs used for training"
)

if __name__ == "__main__":

    args = parser.parse_args()
    print("Les argumentos", args)

    # Experiment epochs
    epochs = args.epochs

    query_samples = 15

    # Number of times to run the experiment
    experiment_times = 100
    all_shots = args.shots

    # Run experiment collecting metrics
    all_metrics = []
    in_dataset, out_dataset = load_data_NLP()

    dataset = "wikitext"

    for shots in [all_shots]:
        experiment_name = "experiments/fine_tunning_{}.csv".format(dataset)

        # Write experiment header
        with open(experiment_name, "a+") as out:
            out.write("dataset, epochs, shots, query_samples, n_experiments\n")
            out.write(
                "{}, {}, {}, {}, {}\n".format(
                    dataset, epochs, shots, query_samples, experiment_times,
                )
            )
            out.write("temp, auc, acc, precision, recall, case_b, case_a\n")

        tmp_metrics = []
        pbar = tqdm(range(experiment_times))
        for i in pbar:

            # Load pretrained model with its training and non-training dataset
            base_model = load_train_model_NLP()

            # Take a random sample to create support and query dataset (training and test, respectively)
            (
                (support_data, support_labels),
                (val_data, val_labels),
                (query_data, query_labels),
            ) = random_sample_wikited_dataset(
                in_dataset, out_dataset, shots, query_samples
            )

            model = Few_shot_model(base_model, support_data, support_labels)

            temp = model.train(
                epochs, query_data, val_data, val_labels, verbose=False
            )

            # Retrieve metrics
            auc, acc, precision, recall, case_b, case_a = compute_metrics(
                model, query_data, query_labels, val_data, val_labels
            )

            all_metrics.append([temp, auc, acc, precision, recall, case_b, case_a])
            tmp_metrics.append([temp, auc, acc, precision, recall, case_b, case_a])
            
            tmp_avg = np.mean(all_metrics, axis=0)

            pbar.set_postfix(
                {
                    "temp": tmp_avg[0],
                    "avg_case_b": tmp_avg[-2],
                    "avg_case_a": tmp_avg[-1],
                }
            )

            # tf.keras.backend.clear_session()
            # del base_model
            # del model
            # gc.collect()

        # write averaged results
        averaged_metrics = np.mean(all_metrics, axis=0)
        error = 1.96 * np.std(all_metrics, ddof=1, axis=0) / np.sqrt(experiment_times)
        with open(experiment_name, "a+") as out:
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(
                    *averaged_metrics
                )
            )
            out.write(
                "{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f},{:.4f}\n".format(*error)
            )
