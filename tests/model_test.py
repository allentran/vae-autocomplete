import unittest

import numpy as np
from vae_seq.model import vae


class VAEModelTest(unittest.TestCase):

    def setUp(self):

        self.output_size = 7
        self.meta_size = 5

        self.model = vae.VAELasagneModel(self.output_size, self.meta_size, samples=2, depth=1)
        self.batch_size = 32

    def train_test(self):

        EPOCHS = 10

        xs = np.random.normal(size=(32, self.output_size)).astype('float32')
        ms = np.random.normal(size=(32, self.meta_size)).astype('float32')
        ys = np.random.normal(size=(32, self.output_size)).astype('float32')
        mask = np.random.randint(0, 2, size=(32, self.output_size)).astype('float32')

        losses = []
        for _ in xrange(EPOCHS):
            losses.append(self.model.fit(xs, ms, ys, mask))

        self.assertLess(losses[-1], losses[0])

    def predict_test(self):

        xs = np.random.normal(size=(32, self.output_size)).astype('float32')
        ms = np.random.normal(size=(32, self.meta_size)).astype('float32')
        mask = np.random.randint(0, 2, size=(32, self.output_size)).astype('float32')

        xhat, logvar = self.model._predict_fn(xs, ms, mask)

        self.assertTrue(xhat.shape, (32, self.output_size, self.model.samples))
        np.testing.assert_array_equal(xhat[:, :, 0], xhat[:, :, 1])

        self.assertTrue(logvar.shape, (32, self.output_size, self.model.samples))
        np.testing.assert_array_equal(logvar[:, :, 0], logvar[:, :, 1])

