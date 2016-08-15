import unittest

import numpy as np
from vae_seq.model import vae


class VAEModelTest(unittest.TestCase):

    def setUp(self):

        self.input_size = 7

        self.model = vae.VAEModel(self.input_size, samples=2)
        self.batch_size = 32

    def train_test(self):

        EPOCHS = 10

        xs = np.random.normal(size=(32, self.input_size)).astype('float32')
        ys = np.random.normal(size=(32, self.input_size)).astype('float32')

        losses = []
        for _ in xrange(EPOCHS):
            losses.append(self.model.train_fn(xs, ys))

        self.assertLess(losses[-1], losses[0])

    def predict_test(self):

        xs = np.random.normal(size=(32, self.input_size)).astype('float32')

        xhat, logvar = self.model.predict_fn(xs)

        self.assertTrue(xhat.shape, (32, self.input_size, self.model.samples))
        np.testing.assert_array_equal(xhat[:, :, 0], xhat[:, :, 1])

        self.assertTrue(logvar.shape, (32, self.input_size, self.model.samples))
        np.testing.assert_array_equal(logvar[:, :, 0], logvar[:, :, 1])

