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

        self.X = np.log(array) - np.log(array[:, 0])[:, None]
        self.X[~np.isfinite(self.X)] = np.nanmean(self.X[np.isfinite(self.X)])
        self.model = vae.VAELasagneModel(self.X.shape[1], self.meta.shape[1], samples=20, depth=3)

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

        def cut_and_get_mask(x, min_cut=2, max_cut=11, force_cut=5):
            if not force_cut:
                cut = np.random.randint(min_cut, max_cut)
            else:
                cut = force_cut
            mask = np.ones(x.shape)
            mask[:, cut:] = 0.
            x_cut = x.copy()
            x_cut[:, cut:] = 0
            return x_cut, mask

        X_train, X_test, meta_train, meta_test = self.split_data(stratify=self.meta[:, 1])
        n_epochs = 0
        epochs_since_min = 0
        min_loss = 1e10
        while True:
            n_epochs += 1
            training_loss = []
            for x_batch, meta_batch in AutocompleteVAE.iterate_minibatches(X_train, meta_train, batchsize=batchsize, shuffle=True):
                x_cut, mask = cut_and_get_mask(x_batch)
                training_loss.append(float(self.model.fit(x_cut, meta_batch, x_batch, mask)))
            x_cut, mask = cut_and_get_mask(X_test, force_cut=5)
            test_loss = self.model.loss(x_cut, meta_test, X_test, mask)

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
