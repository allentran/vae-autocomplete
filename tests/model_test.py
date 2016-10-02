import unittest

import numpy as np
from vae_seq.model import vae


class VAEModelTest(unittest.TestCase):

    def setUp(self):

        self.output_size = 1
        self.meta_size = 5
        self.n_seq = 7

        self.model = vae.EncoderDecodeCompleterModel(self.n_seq, self.output_size, self.meta_size, samples=2, depth=1)
        self.batch_size = 32

    def train_test(self):

        EPOCHS = 10
        cut = 4

        xs = np.random.normal(size=(32, cut, self.output_size)).astype('float32')
        ms = np.random.normal(size=(32, cut, self.meta_size)).astype('float32')
        ys = np.random.normal(size=(32, self.n_seq)).astype('float32')

        losses = []
        for _ in xrange(EPOCHS):
            losses.append(self.model.fit(xs, ms, ys, cut))

        self.assertLess(losses[-1], losses[0])

    def predict_test(self):

        cut = 4

        xs = np.random.normal(size=(32, cut, self.output_size)).astype('float32')
        ms = np.random.normal(size=(32, cut, self.meta_size)).astype('float32')

        xhat, logvar = self.model._predict_fn(xs, ms)

        self.assertTrue(xhat.shape, (32, self.n_seq, self.model.samples))
        np.testing.assert_array_equal(xhat[:, :, 0], xhat[:, :, 1])

        self.assertTrue(logvar.shape, (32, self.n_seq, self.model.samples))
        np.testing.assert_array_equal(logvar[:, :, 0], logvar[:, :, 1])

