from jukebox.hparams import setup_hparams, Hyperparams
from jukebox.make_models import make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi


sample_options = {
    "name": "sample_5b",
    "levels": 3,
    "sample_length_in_seconds": 20,
    "total_sample_length_in_seconds": 180,
    "sr": 44100,
    "n_samples": 6,
    "hop_fraction": [0.5, 0.5, 0.125]
}

rank, local_rank, device = setup_dist_from_mpi(port=29500)
print("Device: {}".format(device))

# hps = Hyperparams(**sample_options)
# hps = setup_hparams("vqvae", dict(sample_length=hps.get('sample_length', 0),
#                                   sample_length_in_seconds=hps.get('sample_length_in_seconds', 0)))
# vqvae = make_vqvae(hps, 'cuda:0')