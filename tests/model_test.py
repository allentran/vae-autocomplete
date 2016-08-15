import unittest

import numpy as np
from vae_seq.model import vae


class VAEModelTest(unittest.TestCase):

    def setUp(self):

        self.input_size = 7

        self.model = vae.VAEModel(self.input_size, samples=2)
        self.batch_size = 32

    def train_test(self):

        xs = np.random.normal(size=(32, self.input_size)).astype('float32')
        ys = np.random.normal(size=(32, self.input_size)).astype('float32')

        losses = []
        for _ in xrange(10):
            losses.append(self.model.train_fn(xs, ys))

        self.assertLess(losses[-1], losses[0])



