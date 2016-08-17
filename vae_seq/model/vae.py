

import theano
import theano.tensor as TT
import numpy as np
from lasagne import layers, regularization, nonlinearities, updates

import sampling


class VAEModel(object):

    def __init__(self, output_size, meta_size, depth=2, samples=10):

        def neg_log_likelihood(q_mu, q_log_var, y, mu_x, logvar_x):

            KL_loss = -0.5 * TT.sum(1 + q_log_var - TT.square(q_mu) - TT.exp(q_log_var), axis=-1)
            reconstruction_loss = np.float32(0.5 * np.log(2. * np.pi)) + 0.5 * logvar_x + (TT.square(y[:, :, None] - mu_x)) / TT.exp(2. * logvar_x)
            return reconstruction_loss.mean(axis=-1).sum(axis=1) + KL_loss

        self.samples = samples

        encoder_sizes = [40, 20]
        decoder_sizes = [30, 30]
        latent_size = 13

        input_var = TT.matrix()
        target_var = TT.matrix()

        input_layer = layers.InputLayer((None, output_size + meta_size), input_var=input_var)
        dense = input_layer

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
            layers.get_output(mu),
            layers.get_output(log_var),
            target_var,
            layers.get_output(mu_x_layer),
            layers.get_output(logvar_x_layer)
        ).mean()

        test_loss = neg_log_likelihood(
            layers.get_output(mu, deterministic=True),
            layers.get_output(log_var, deterministic=True),
            target_var,
            layers.get_output(mu_x_layer, deterministic=True),
            layers.get_output(logvar_x_layer, deterministic=True)
        ).mean()

        params = layers.get_all_params(mu_and_logvar_x_layer, trainable=True)
        param_updates = updates.adadelta(loss, params)

        self._train_fn = theano.function(
            [input_var, target_var],
            updates=param_updates,
            outputs=loss
        )

        self._loss_fn = theano.function(
            [input_var, target_var],
            outputs=test_loss
        )

        self.latent_output_fn = theano.function(
            [input_var],
            outputs=layers.get_output(z_x, deterministic=True)
        )

        self._predict_fn = theano.function(
            [input_var],
            outputs=[
                layers.get_output(mu_x_layer, deterministic=True),
                layers.get_output(logvar_x_layer, deterministic=True)
            ]
        )

    def fit(self, x_matrix, y_matrix):
        x = x_matrix.astype('float32')
        y = y_matrix.astype('float32')
        return self._train_fn(x, y)

    def predict(self, x_matrix):
        x = x_matrix.astype('float32')
        mu, logvar = self._predict_fn(x)
        return mu[:, :, 0], np.exp(logvar[:, :, 0])

    def loss(self, x_matrix, y_matrix):
        x = x_matrix.astype('float32')
        y = y_matrix.astype('float32')
        return self._loss_fn(x, y)



