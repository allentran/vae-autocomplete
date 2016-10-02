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

        self.model = vae.EncoderDecodeCompleterModel(self.X.shape[1], 1, self.meta.shape[1], samples=2, depth=3)

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

        def cut(x, min_cut=2, max_cut=11, force_cut=5):
            if not force_cut:
                cut = np.random.randint(min_cut, max_cut)
            else:
                cut = force_cut
            return x.copy()[:, :cut], cut

        X_train, X_test, meta_train, meta_test = self.split_data(stratify=self.meta[:, 1])
        n_epochs = 0
        epochs_since_min = 0
        min_loss = 1e10
        while True:
            n_epochs += 1
            training_loss = []
            for x_batch, meta_batch in AutocompleteVAE.iterate_minibatches(X_train, meta_train, batchsize=batchsize, shuffle=True):
                x_cut, n_seq = cut(x_batch)
                training_loss.append(
                    self.model.fit(
                        x_cut[:, :, None],
                        np.repeat(meta_batch[:, None, :], n_seq, axis=1),
                        x_batch,
                        n_seq
                    )
                )
            x_cut, n_seq = cut(X_test, force_cut=5)
            test_loss = self.model.loss(
                x_cut[:, :, None],
                np.repeat(meta_test[:, None, :], n_seq, axis=1),
                X_test,
                n_seq
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

        import IPython
        IPython.embed()
