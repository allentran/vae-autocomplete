import numpy as np
import theano
import theano.tensor as TT
from lasagne import layers, nonlinearities, updates, regularization

import sampling
from vae_seq.model.layers import neg_log_likelihood, neg_log_likelihood2, kl_loss


class EncoderDecodeCompleterModel(object):

    def __init__(self, full_length, output_size, meta_size, depth=2, samples=10, encoder_size=64, decoder_size=64):

        self.samples = samples

        latent_size = 8

        input_var = TT.tensor3(dtype='float32')
        meta_var = TT.tensor3(dtype='float32')
        target_var = TT.matrix()
        cut_var = TT.scalar(dtype='int32')

        input_layer = layers.InputLayer((None, None, output_size), input_var=input_var)
        meta_layer = layers.InputLayer((None, None, meta_size), input_var=meta_var)
        meta_layer = layers.DropoutLayer(meta_layer, p=0.2)
        concat_input_layer = layers.ConcatLayer([input_layer, meta_layer], axis=-1)

        # encoder
        lstm_layer = layers.RecurrentLayer(concat_input_layer, encoder_size / 2, learn_init=True)
        lstm_layer = layers.RecurrentLayer(lstm_layer, encoder_size / 2, only_return_final=True, learn_init=True)

        encoded = layers.DenseLayer(lstm_layer, latent_size)
        encoded = layers.batch_norm(encoded)

        # decoder
        dense = encoded
        for idx in xrange(depth):
            dense = layers.DenseLayer(dense, decoder_size)
            dense = layers.batch_norm(dense)

        mu_and_logvar_x_layer = layers.DenseLayer(dense, full_length * 2, nonlinearity=nonlinearities.linear)

        mu_x_layer = layers.SliceLayer(mu_and_logvar_x_layer, slice(0, full_length), axis=1)
        logvar_x_layer = layers.SliceLayer(mu_and_logvar_x_layer, slice(full_length, None), axis=1)

        l2_norm = regularization.regularize_network_params(mu_and_logvar_x_layer, regularization.l2)

        loss = neg_log_likelihood(
            target_var,
            layers.get_output(mu_x_layer, deterministic=False),
            layers.get_output(logvar_x_layer, deterministic=False),
            cut_var
        ) + 1e-4 * l2_norm

        test_loss = neg_log_likelihood(
            target_var,
            layers.get_output(mu_x_layer, deterministic=True),
            layers.get_output(logvar_x_layer, deterministic=True),
            cut_var
        )

        params = layers.get_all_params(mu_and_logvar_x_layer, trainable=True)
        param_updates = updates.adadelta(loss.mean(), params)

        self._train_fn = theano.function(
            [input_var, meta_var, target_var, cut_var],
            updates=param_updates,
            outputs=loss.mean()
        )

        self._loss_fn = theano.function(
            [input_var, meta_var, target_var, cut_var],
            outputs=test_loss.mean()
        )

        self._predict_fn = theano.function(
            [input_var, meta_var],
            outputs=[
                layers.get_output(mu_x_layer, deterministic=True),
                layers.get_output(logvar_x_layer, deterministic=True)
            ]
        )

    def fit(self, x_matrix, meta_matrix, y_matrix, cut_var):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        y = y_matrix.astype('float32')

        return self._train_fn(x, m, y, cut_var)

    def predict(self, x_matrix, meta_matrix):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')

        mu, logvar = self._predict_fn(x, m)
        return mu, np.exp(logvar)

    def loss(self, x_matrix, meta_matrix, y_matrix, cut_var):
        x = x_matrix.astype('float32')
        m = meta_matrix.astype('float32')
        y = y_matrix.astype('float32')

        return self._loss_fn(x, m, y, cut_var)


class BaselineFeedForwardModel(object):

    def __init__(self, output_size, meta_size, depth=2):

        encoder_sizes = [64, 64, 64]

        input_var = TT.matrix()
        meta_var = TT.matrix()
        target_var = TT.matrix()
        mask_var = TT.matrix()

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
            layers.get_output(log_var),
            mask_var
        ).mean()

        test_loss = neg_log_likelihood2(
            target_var,
            layers.get_output(mu, deterministic=True),
            layers.get_output(log_var, deterministic=True),
            mask_var
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
