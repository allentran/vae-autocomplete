import logging

import numpy as np
from sklearn.cross_validation import train_test_split

from vae_seq.model import vae

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class AutocompleteVAE(object):

    def __init__(self, array, meta, standardize_meta=True, diff=True):

        assert array.shape[0] == meta.shape[0]

        if standardize_meta:
            self.meta_means = np.mean(meta, axis=0, keepdims=True)
            self.meta_stds = np.std(meta, axis=0, keepdims=True)
            self.meta = (meta - self.meta_means) / self.meta_stds
        else:
            self.meta = meta

        self.standardize_meta = standardize_meta
        self.X = np.log(array[:, 1:]) - np.log(array[:, 0:-1])
        drop_mask = (~np.isfinite(self.X)).sum(axis=1) > 0
        self.X = self.X[~drop_mask, :]
        self.meta = self.meta[~drop_mask, :]
        self.n_seq = self.X.shape[1]

        self.model = vae.EncoderDecodeCompleterModel(self.n_seq, 1, self.meta.shape[1], depth=3)

    def split_data(self, test_frac=0.2, stratify=None):
        return train_test_split(self.X, self.meta, test_size=test_frac, stratify=stratify)

    @staticmethod
    def iterate_minibatches(inputs, meta, batchsize, shuffle=True):
        assert len(inputs) == len(meta)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], meta[excerpt]

    def fit(self, epochs=None, epochs_since_best=10, batchsize=100):

        X_train, X_test, meta_train, meta_test = self.split_data(stratify=self.meta[:, 1])
        n_epochs = 0
        epochs_since_min = 0
        min_loss = 1e10
        weights = np.ones(self.n_seq)
        while True:
            n_epochs += 1
            training_loss = []
            for x_batch, meta_batch in AutocompleteVAE.iterate_minibatches(X_train, meta_train, batchsize=batchsize, shuffle=True):
                training_loss.append(
                    self.model.fit(
                        x_batch[:, :, None],
                        np.repeat(meta_batch[:, None, :], self.n_seq, axis=1),
                        x_batch,
                        weights
                    )
                )
            test_loss = self.model.loss(
                X_test[:, :, None],
                np.repeat(meta_test[:, None, :], self.n_seq, axis=1),
                X_test,
                weights
            )

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

        yhat, yvar = self.model.predict(X_test[:, :, None], np.repeat(meta_test[:, None, :], self.n_seq, axis=1))
        corr = np.corrcoef(yhat[:, 0, :].flatten(), X_test[:, :].flatten())[0, 1]
        logger.info("Correlation after a single observation: %.2f" % corr)
