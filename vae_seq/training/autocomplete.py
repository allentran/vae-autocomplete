import logging

import numpy as np
from sklearn.cross_validation import train_test_split

from ..model import vae

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class AutocompleteVAE(object):

    def __init__(self, array, meta, standardize_meta=True):

        assert array.shape[0] == meta.shape[0]

        if standardize_meta:
            self.meta_means = np.mean(meta, axis=0, keepdims=True)
            self.meta_stds = np.std(meta, axis=0, keepdims=True)
            self.meta = (meta - self.meta_means) / self.meta_stds
        else:
            self.meta = meta

        self.standardize_meta = standardize_meta

        self.raw_array = array
        self.X_means = np.mean(array, axis=0, keepdims=True)
        self.X_stds = np.std(array, axis=0, keepdims=True)

        self.X = (array - self.X_means) / self.X_stds

        self.model = vae.VAEModel(self.X.shape[1], self.meta.shape[1])

    def cleave_data(self, last_idx):

        self.Y = self.X.copy()
        self.X[:, last_idx + 1:] = 0

    def split_data(self, test_frac=0.2):

        X = np.hstack((self.meta, self.X))
        return train_test_split(X, self.Y, test_size=test_frac)

    @staticmethod
    def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    def fit(self, epochs=None, epochs_since_best=10, batchsize=128):

        X_train, X_test, y_train, y_test = self.split_data()

        n_epochs = 0
        epochs_since_min = 0
        min_loss = 1e10
        while True:
            n_epochs += 1
            training_loss = []
            for x_batch, y_batch in AutocompleteVAE.iterate_minibatches(X_train, y_train, batchsize=batchsize):
                training_loss += self.model.fit(x_batch, y_batch)
            test_loss = self.model.loss(X_test, y_test)

            logger.info("%s epoch: training=%.2f, test=%.2f", n_epochs, np.mean(training_loss), test_loss)

            if epochs:
                if n_epochs > epochs:
                    break
            else:
                if test_loss < min_loss:
                    min_loss = test_loss
                    epochs_since_min = 0
                else:
                    epochs_since_min += 1
                    if epochs_since_min > epochs_since_best:
                        break
