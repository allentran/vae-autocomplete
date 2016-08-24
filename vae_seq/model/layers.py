from theano import tensor as TT
from theano.tensor.shared_randomstreams import RandomStreams
from lasagne import layers


def neg_log_likelihood(y, mu_x, logvar_x):
    reconstruction_loss = 0.5 * logvar_x + 0.5 * (TT.square(y[:, :, None] - mu_x)) / TT.exp(logvar_x)
    return reconstruction_loss.mean(axis=-1).sum(axis=1)


def neg_log_likelihood2(y, mu_x, logvar_x):
    reconstruction_loss = 0.5 * logvar_x + 0.5 * (TT.square(y - mu_x)) / TT.exp(logvar_x)
    return reconstruction_loss.sum(axis=1)


def kl_loss(q_mu, q_log_var):
    return -0.5 * TT.sum(1 + q_log_var - TT.square(q_mu) - TT.exp(q_log_var), axis=-1)


class RandomZeroOutLayer(layers.Layer):
    def __init__(self, incoming, min_cutoff, max_cutoff, name=None):
        super(RandomZeroOutLayer, self).__init__(incoming, name)
        self.min_cutoff = min_cutoff
        self.max_cutoff = max_cutoff
        self.srng = RandomStreams(seed=1692)

    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            return input
        rnd_index = self.srng.random_integers(low=self.min_cutoff, high=self.max_cutoff)
        return TT.set_subtensor(input[:, rnd_index:], 0)



