import numpy as np
import theano
import theano.tensor as TT
from lasagne import layers, nonlinearities, updates

import sampling
from vae_seq.model.layers import neg_log_likelihood, neg_log_likelihood2, kl_loss, RandomZeroOutLayer


class VAELasagneModel(object):

    def __init__(self, output_size, meta_size, depth=2, samples=10):

        self.samples = samples

        encoder_sizes = [64, 64, 64]
        decoder_sizes = [64, 64, 64]
        latent_size = 4

        input_var = TT.matrix()
        meta_var = TT.matrix()
        target_var = TT.matrix()

        input_layer = layers.InputLayer((None, output_size), input_var=input_var)
        input_layer = RandomZeroOutLayer(input_layer, 3, 6)
        meta_layer = layers.InputLayer((None, meta_size), input_var=meta_var)
        concat_input_layer = layers.ConcatLayer([input_layer, meta_layer])
        dense = concat_input_layer

        # encoder
        for idx in xrange(depth):
            dense = layers.DenseLayer(dense, encoder_sizes[idx])
            dense = layers.batch_norm(dense)

        mu = layers.DenseLayer(dense, latent_size)
        log_var = layers.DenseLayer(dense, latent_size)

        z_x = sampling.GaussianSamplerLayer(mu, log_var, samples) # returns batch x latent_size x samples

        # decoder
        dense = layers.DimshuffleLayer(z_x, (0, 2, 1)) # batch x samples x latent
        dense = layers.ReshapeLayer(dense, (-1, dense.output_shape[2])) # batch . samples x latent
        dense = dense
        for idx in xrange(depth):
            dense = layers.DenseLayer(dense, decoder_sizes[idx])
            dense = layers.batch_norm(dense)

        mu_and_logvar_x_layer = layers.DenseLayer(dense, output_size * 2, nonlinearity=nonlinearities.linear)

        mu_and_logvar_x_layer = layers.ReshapeLayer(mu_and_logvar_x_layer, (-1, z_x.samples, mu_and_logvar_x_layer.output_shape[1]))
        mu_and_logvar_x_layer = layers.DimshuffleLayer(mu_and_logvar_x_layer, (0, 2, 1))

        mu_x_layer = layers.SliceLayer(mu_and_logvar_x_layer, slice(0, output_size), axis=1)
        logvar_x_layer = layers.SliceLayer(mu_and_logvar_x_layer, slice(output_size, None), axis=1)

        loss = neg_log_likelihood(
            target_var,
            layers.get_output(mu_x_layer),
            layers.get_output(logvar_x_layer)
        ) + kl_loss(layers.get_output(mu), layers.get_output(log_var))

        test_loss = neg_log_likelihood(
            target_var,
            layers.get_output(mu_x_layer, deterministic=True),
            layers.get_output(logvar_x_layer, deterministic=True),
        ) + kl_loss(layers.get_output(mu, deterministic=True), layers.get_output(log_var, deterministic=True))

        params = layers.get_all_params(mu_and_logvar_x_layer, trainable=True)
        param_updates = updates.adadelta(loss.mean(), params)

        self._train_fn = theano.function(
            [input_var, meta_var, target_var],
            updates=param_updates,
            outputs=loss.mean()
        )

        self._loss_fn = theano.function(
            [input_var, meta_var, target_var],
            outputs=test_loss.mean()
        )

        self._predict_fn = theano.function(
            [input_var, meta_var],
            outputs=[
                layers.get_output(mu_x_layer, deterministic=True),
                layers.get_output(logvar_x_layer, deterministic=True)
            ]
        )

    def fit(self, x_matrix, meta_matrix, y_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        y = y_matrix.astype('float32')
        return self._train_fn(x, m, y)

    def predict(self, x_matrix, meta_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        mu, logvar = self._predict_fn(x, m)
        return mu[:, :, 0], np.exp(logvar[:, :, 0])

    def loss(self, x_matrix, meta_matrix, y_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        y = y_matrix.astype('float32')
        return self._loss_fn(x, m, y)


class BaselineFeedForwardModel(object):

    def __init__(self, output_size, meta_size, depth=2):

        encoder_sizes = [64, 64, 64]

        input_var = TT.matrix()
        meta_var = TT.matrix()
        target_var = TT.matrix()

        input_layer = layers.InputLayer((None, output_size), input_var=input_var)
        meta_layer = layers.InputLayer((None, meta_size), input_var=meta_var)
        concat_input_layer = layers.ConcatLayer([input_layer, meta_layer])
        dense = concat_input_layer

        for idx in xrange(depth):
            dense = layers.DenseLayer(dense, encoder_sizes[idx])
            dense = layers.batch_norm(dense)

        mu_and_logvar = layers.DenseLayer(dense, 2 * output_size, nonlinearity=nonlinearities.linear)
        mu = layers.SliceLayer(mu_and_logvar, slice(0, output_size), axis=1)
        log_var = layers.SliceLayer(mu_and_logvar, slice(output_size, None), axis=1)

        loss = neg_log_likelihood2(
            target_var,
            layers.get_output(mu),
            layers.get_output(log_var)
        ).mean()

        test_loss = neg_log_likelihood2(
            target_var,
            layers.get_output(mu, deterministic=True),
            layers.get_output(log_var, deterministic=True),
        ).mean()

        params = layers.get_all_params(mu_and_logvar, trainable=True)
        param_updates = updates.adadelta(loss, params)

        self._train_fn = theano.function(
            [input_var, meta_var, target_var],
            updates=param_updates,
            outputs=loss
        )

        self._loss_fn = theano.function(
            [input_var, meta_var, target_var],
            outputs=test_loss
        )

        self._predict_fn = theano.function(
            [input_var, meta_var],
            outputs=[
                layers.get_output(mu, deterministic=True),
                layers.get_output(log_var, deterministic=True)
            ]
        )

    def fit(self, x_matrix, meta_matrix, y_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        y = y_matrix.astype('float32')
        return self._train_fn(x, m, y)

    def predict(self, x_matrix, meta_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        mu, logvar = self._predict_fn(x, m)
        return mu, np.exp(logvar)

    def loss(self, x_matrix, meta_matrix, y_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        y = y_matrix.astype('float32')
        return self._loss_fn(x, m, y)



