from theano import tensor as TT


def neg_log_likelihood(y, mu_x, logvar_x, cut_weights):
    clipped_logvar = TT.clip(logvar_x, -5, 5)
    reconstruction_loss = 0.5 * clipped_logvar + 0.5 * (TT.square(y[:, None, :] - mu_x)) / TT.exp(clipped_logvar)
    loss_per_seq_upto_t = reconstruction_loss.sum(axis=2)
    loss_per_seq = cut_weights[None, :] * loss_per_seq_upto_t
    return loss_per_seq.sum(axis=1)


def neg_log_likelihood2(y, mu_x, logvar_x):
    reconstruction_loss = 0.5 * logvar_x + 0.5 * (TT.square(y - mu_x)) / TT.exp(logvar_x)
    return reconstruction_loss.sum(axis=1)


def kl_loss(q_mu, q_log_var):
    return -0.5 * TT.sum(1 + q_log_var - TT.square(q_mu) - TT.exp(q_log_var), axis=-1)
