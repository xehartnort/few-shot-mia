import tensorflow as tf
import keras_wrn
import numpy as np
import faiss
from transformers import TFGPT2LMHeadModel as gpt2, GPT2Config
from datasets import load_from_disk
from transformers import DefaultDataCollator
from transformers.utils.logging import set_verbosity_error
# Compute metrics
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
)

set_verbosity_error()

# boilerplate to use arguments in the dense layer initializer
class SupportInit(tf.keras.initializers.Initializer):
    def __init__(self, model, data, labels):
        self._model = model
        self._data = data
        self._labels = np.argmax(labels, axis=1)
        self._sample = []

        for i in [0, 1]:
            selected_data = self._data[self._labels == i, :]
            logits = self._model(selected_data)
            logits = tf.nn.relu(logits)  # apply relu activation
            logits = np.mean(logits, axis=0)  # average
            logits = logits / np.maximum(np.linalg.norm(logits), 1e-6)  # normalize
            self._sample.append(logits)
        self._sample = np.array(self._sample)

        self._sample = np.transpose(self._sample)

    def __call__(self, shape, dtype=None):
        return tf.Variable(self._sample, dtype=dtype)


# boilerplate to use arguments in the dense layer initializer
class SupportInitNLP(tf.keras.initializers.Initializer):
    def __init__(self, model, data, labels):
        self._model = model
        self._data = data
        self._labels = np.argmax(labels, axis=1)
        self._sample = []

        for i in [0, 1]:
            selected_data = self._data[self._labels == i, :]
            # logits = self._model(selected_data).logits
            logits = self._model(selected_data).logits[:,:,-1,:]
            # breakpoint()
            # logits = tf.keras.layers.Flatten()(logits)
            # logits = tf.nn.softmax(logits, axis=2).numpy()
            # logits = np.array([i.flatten() for i in logits])
            logits = tf.nn.relu(logits)  # apply relu activation
            logits = np.mean(logits, axis=0)  # average
            logits = logits / np.maximum(np.linalg.norm(logits), 1e-6)  # normalize
            self._sample.append(logits)

        self._sample = np.transpose(np.array(self._sample))

    def __call__(self, shape, dtype=None):
        return tf.Variable(tf.squeeze(self._sample), dtype=dtype)

# Stop training when a certain accuracy is reached
class StopOnPoint(tf.keras.callbacks.Callback):
    def __init__(self, point):
        super(StopOnPoint, self).__init__()
        self.point = point

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs["categorical_accuracy"]
        self.model.stop_training = accuracy >= self.point


# https://github.com/hypnopump/keras-wrn
# Use it to train a WideResNet for Cifar10 and Cifar100
def load_train_model(cifar10=True, k=28, n=2):
    if cifar10:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
        model_name = "models/cifar10_wrn_{}_{}.h5".format(k, n)
    else:
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()
        model_name = "models/cifar100_wrn_{}_{}.h5".format(k, n)

    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    
    try:
        model = tf.keras.models.load_model(model_name)
    except OSError:  # File does not exists
        shape, classes = (32, 32, 3), 10 if cifar10 else 100

        model = keras_wrn.build_model(
            shape, classes, k, n
        )  # 16, 4 was the original, 28, 10 is an alternative
        scheduler = tf.keras.optimizers.schedules.CosineDecay(0.1, 5000)
        opt = tf.keras.optimizers.SGD(learning_rate=scheduler, momentum=0.99, use_ema=True, weight_decay=0.0005)
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.CategoricalAccuracy()]

        model.compile(opt, loss, metrics)

        model.fit(
            x_train[0:25_000],
            y_train[0:25_000],
            epochs=100,
            batch_size=256,
            callbacks=[StopOnPoint(0.6)],
        )

        model.save(model_name)

    in_data = x_train[0:25_000]
    in_labels = y_train[0:25_000]
    out_data = x_train[25_000:]
    out_labels = y_train[25_000:]

    return model, (in_data, in_labels), (out_data, out_labels)


def load_data_NLP():
    out_data = load_from_disk("wikitext/wikitext.no_train")
    in_data = load_from_disk("wikitext/wikitext.train")

    data_collator = DefaultDataCollator(return_tensors="tf")
    options = tf.data.Options()

    out_dataset = (
        out_data.to_tf_dataset(
            # labels are passed as input, as we will use the model's internal loss
            columns='input_ids', #[col for col in out_data.features if col != "special_tokens_mask"],
            batch_size=1,
            shuffle=True,
            collate_fn=data_collator,
            drop_remainder=False,
        )
        .with_options(options)
        .repeat()
    )

    in_dataset = (
        in_data.to_tf_dataset(
            # labels are passed as input, as we will use the model's internal loss
            columns='input_ids', #[col for col in in_data.features if col != "special_tokens_mask"],
            batch_size=1,
            shuffle=True,
            collate_fn=data_collator,
            drop_remainder=False,
        )
        .with_options(options)
        .repeat()
    )

    return in_dataset, out_dataset


def load_train_model_NLP():
    # Initializing a GPT2 configuration
    configuration = GPT2Config()
    # Initializing a model from the configuration
    model = gpt2(configuration)
    model.from_pretrained("wikitext/new_gpt2_model")
    return model


def compute_metrics(fs_model, query_data, query_labels, val_data, val_labels):
    # Get threshold for the validation set and apply it to the query set
    predictions = fs_model(query_data)
    # val_predictions = fs_model(val_data)
    probs = predictions[:, 1]
    # val_probs = val_predictions[:, 1]

    sparse_labels = np.argmax(query_labels, axis=1)
    sparse_preds = np.argmax(predictions, axis=1)

    val_sparse_labels = np.argmax(val_labels, axis=1)
    # val_sparse_preds = np.argmax(val_predictions, axis=1)

    auc = roc_auc_score(sparse_labels, probs)

    fpr, tpr, thresholds = roc_curve(sparse_labels, probs)
    # val_fpr, val_tpr, val_thresholds = roc_curve(val_sparse_labels, val_probs)

    d = len(query_data)
    half_d = d / 2

    # All the false positives
    fps = np.array(fpr) * half_d
    # val_fps = np.array(val_fpr) * half_d
    # All the true positives
    tps = np.array(tpr) * half_d
    # val_tps = np.array(val_tpr) * half_d

    # Case A => our metric at FP = 0
    # case_a_threshold = min(val_thresholds[val_fps <= 0], default=0)
    # tps_at_very_low_fps = min(tps[thresholds <= case_a_threshold], default=0)
    tps_at_very_low_fps = max(tps[fps <= 0], default=0)
    case_a = np.log(tps_at_very_low_fps + 1) / np.log(half_d + 1)

    # Case B => the best of our metric at FP <= ln(half dataset)
    # case_b_threshold = min(val_thresholds[val_fps <= np.ceil(np.log(d))], default=0)
    # tps_at_low_fps = min(tps[thresholds <= case_b_threshold], default=0)
    tps_at_low_fps = max(tps[fps<=np.ceil(np.log(d))])
    case_b = np.log(tps_at_low_fps + 1) / np.log(half_d + 1)

    acc = accuracy_score(sparse_labels, sparse_preds)
    precision = precision_score(sparse_labels, sparse_preds, zero_division=0)
    recall = recall_score(sparse_labels, sparse_preds, zero_division=0)

    return auc, acc, precision, recall, case_b, case_a


class FaissKNeighbors:
    def __init__(self, k=5):
        self.index = None
        self.y = None
        self.k = k

    def fit(self, X, y=None):
        d = X.shape[1]
        self.index = faiss.IndexFlatL2(d)  # Indx
        self.index.train(X.astype(np.float32))
        self.index.add(X.astype(np.float32))
        self.y = y
        return self

    def predict(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        votes = self.y[indices]
        predictions = np.array([np.argmax(np.bincount(x)) for x in votes])
        return predictions

    def predict_proba(self, X):
        preds = []
        distances, indices = self.index.search(X.astype(np.float32), k=self.k)
        for dist, idx in zip(distances, indices):
            votes = self.y[idx]
            out_prob = np.sum((votes == 0) / dist)
            in_prob = np.sum((votes == 1) / dist)
            t = out_prob + in_prob
            preds.append([out_prob / t, in_prob / t])
        return np.array(preds)

    def kneighbors(self, X):
        _, indices = self.index.search(X.astype(np.float32), k=self.k)
        return _, indices

    def distance_matrix(self, X):
        dist, indices = self.index.search(X.astype(np.float32), k=self.k)
        # Cada fila, tiene la distancias de todos los centroides a una feature
        dst_matrix = [dst[np.argsort(idx)] for dst, idx in zip(dist, indices)]
        return np.array(dst_matrix)


def random_sample_datasets(in_data, out_data, shots, query_samples):
    in_indexing = np.random.choice(
        len(in_data), size=shots + 2 * query_samples, replace=False
    )
    out_indexing = np.random.choice(
        len(out_data), size=shots + 2 * query_samples, replace=False
    )

    support_data = np.concatenate(
        (in_data[in_indexing[:shots], :], out_data[out_indexing[:shots], :])
    )
    support_labels = np.concatenate((np.ones(shots), np.zeros(shots)))
    support_labels = tf.keras.utils.to_categorical(support_labels)

    val_data = np.concatenate(
        (
            in_data[in_indexing[shots : shots + query_samples], :],
            out_data[out_indexing[shots : shots + query_samples], :],
        )
    )
    val_labels = np.concatenate((np.ones(query_samples), np.zeros(query_samples)))
    val_labels = tf.keras.utils.to_categorical(val_labels)

    query_data = np.concatenate(
        (
            in_data[in_indexing[shots + query_samples :], :],
            out_data[out_indexing[shots + query_samples :], :],
        )
    )
    query_labels = np.concatenate((np.ones(query_samples), np.zeros(query_samples)))
    query_labels = tf.keras.utils.to_categorical(query_labels)

    return (
        (support_data, support_labels),
        (val_data, val_labels),
        (query_data, query_labels),
    )


def get_n_items(dataset, n):
    return next(dataset.repeat().batch(n).as_numpy_iterator())

def get_slice(dict_data, slices):
    return {
        'labels': dict_data['labels'][slices],
        'inputs_ids': dict_data['input_ids'][slices],
        'attention_mask': dict_data['attention_mask'][slices]
    }

def nlp_concatenate(dict1, dict2):
    return {
        'labels': np.concatenate((dict1['labels'], dict2['labels'])),
        'inputs_ids': np.concatenate((dict1['inputs_ids'], dict2['inputs_ids'])),
        'attention_mask': np.concatenate((dict1['attention_mask'], dict2['attention_mask'])),
    }

def random_sample_wikited_dataset(in_data, out_data, shots, query_samples):
    in_ = get_n_items(in_data, shots + query_samples * 2)
    out_ = get_n_items(out_data, shots + query_samples * 2)

    #nlp_concatenate(get_slice(in_, custom_slice), get_slice(out_, custom_slice))
    custom_slice = slice(None, shots)
    support_data = np.concatenate((in_[custom_slice], out_[custom_slice]))
    support_labels = np.concatenate((np.ones(shots), np.zeros(shots)))
    support_labels = tf.keras.utils.to_categorical(support_labels)

    custom_slice = slice(shots, shots+query_samples)
    val_data = np.concatenate((in_[custom_slice], out_[custom_slice]))
    val_labels = np.concatenate((np.ones(query_samples), np.zeros(query_samples)))
    val_labels = tf.keras.utils.to_categorical(val_labels)

    custom_slice = slice(shots + query_samples, None)
    query_data = np.concatenate((in_[custom_slice], out_[custom_slice]))
    query_labels = np.concatenate((np.ones(query_samples), np.zeros(query_samples)))
    query_labels = tf.keras.utils.to_categorical(query_labels)

    return (
        (support_data, support_labels),
        (val_data, val_labels),
        (query_data, query_labels),
    )
