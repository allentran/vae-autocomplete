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
            self.meta_stds[self.meta_stds == 0] = 1e-10
            self.meta = self._standardize_meta(meta)
        else:
            self.meta = meta

        self.standardize_meta = standardize_meta
        self.diff = diff

        if diff:
            self.X, self.meta = self._log_diff(array, self.meta)
        else:
            self.X = array

        self.n_seq = self.X.shape[1]
        self.max_cut = self.X.shape[1]
        self.min_cut = self.max_cut / 2

        self.model = vae.EncoderDecodeCompleterModel(self.X.shape[1], 1, self.meta.shape[1], depth=3)

    def split_data(self, test_frac=0.2, stratify=None):
        return train_test_split(self.X, self.meta, test_size=test_frac, stratify=stratify)

    def _standardize_meta(self, meta):
        return (meta - self.meta_means) / self.meta_stds

    def _log_diff(self, array, meta, drop = True):
        X = np.log(array[:, 1:]) - np.log(array[:, 0:-1])
        if drop:
            drop_mask = (~np.isfinite(X)).sum(axis=1) > 0
            X = X[~drop_mask, :]
            meta = meta[~drop_mask, :]
        return X, meta

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

    def predict(self, X, meta, filled_length):
        if self.standardize_meta:
            meta = self._standardize_meta(meta)

        if self.diff:
            X, meta = self._log_diff(X, meta, drop=False)

        if filled_length <= 0:
            X[:, 0] = np.log(1) - np.log(1)

        result = self.model.predict(
            X[:, :, None],
            np.repeat(meta[:, None, :], self.n_seq, axis=1)
        )

        finite_index = max(
            [i for i in range(self.n_seq) if all(np.isfinite(result[0][:, i, :].flatten()))]
        )

        return result[0][:, finite_index, :], result[1][:, finite_index, :]

    def _cut(self, x, force_cut=None):
        if not force_cut:
            cut = np.random.randint(self.min_cut, self.max_cut)
        else:
            cut = force_cut
        return x.copy()[:, :cut], cut

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
