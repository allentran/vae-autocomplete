

import numpy as np
from vae_seq.training.autocomplete import AutocompleteVAE


def main():

    sequences = np.loadtxt('paper/seqs.npy')
    meta = np.loadtxt('paper/metas.npy')

    vae = AutocompleteVAE(sequences, meta)
    vae.fit(epochs=150)

if __name__ == "__main__":
    main()
