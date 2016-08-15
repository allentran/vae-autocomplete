import theano.tensor as TT
from theano.tensor import shared_randomstreams
from lasagne import layers


class GaussianSamplerLayer(layers.MergeLayer):
    def __init__(self, mu, logvar, samples=10, **kwargs):
        self.rng = shared_randomstreams.RandomStreams(1692)
        self.samples = samples
        super(GaussianSamplerLayer, self).__init__([mu, logvar], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0] + (self.samples, )

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, log_var = inputs
        if deterministic:
            return TT.repeat(mu[:, :, None], self.samples, axis=2)
        shape=(
                  self.input_shapes[0][0] or inputs[0].shape[0],
                  self.input_shapes[0][1] or inputs[0].shape[1]
              ) + (self.samples, )
        return mu[:, :, None] + TT.exp(log_var / 2.)[:, :, None] * self.rng.normal(shape)
