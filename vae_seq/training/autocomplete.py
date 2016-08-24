import logging

import numpy as np
from sklearn.cross_validation import train_test_split

from vae_seq.model import vae

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

        self.X = array
        self.model = vae.VAELasagneModel(self.X.shape[1], self.meta.shape[1], depth=3)

    def cleave_data(self, last_idx):
        Y = self.X.copy()
        Y[:, last_idx + 1:] = 0
        return Y

    def split_data(self, test_frac=0.2):
        noisy_X = self.cleave_data(3)
        return train_test_split(noisy_X, self.meta, self.X, test_size=test_frac)

    @staticmethod
    def iterate_minibatches(inputs, meta, targets, batchsize, shuffle=True):
        assert len(inputs) == len(targets) == len(meta)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], meta[excerpt], targets[excerpt]

    def fit(self, epochs=None, epochs_since_best=10, batchsize=100):

        X_train, X_test, meta_train, meta_test, Y_train, Y_test = self.split_data()

        n_epochs = 0
        epochs_since_min = 0
        min_loss = 1e10
        while True:
            n_epochs += 1
            training_loss = []
            for x_batch, meta_batch, y_batch in AutocompleteVAE.iterate_minibatches(X_train, meta_train, Y_train, batchsize=batchsize):
                training_loss.append(float(self.model.fit(x_batch, meta_batch, y_batch)))
            test_loss = self.model.loss(X_test, meta_test, Y_test)

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
