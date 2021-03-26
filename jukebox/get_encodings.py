from jukebox.hparams import setup_hparams, Hyperparams
from jukebox.make_models import make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.data.data_processor import DataProcessor
import jukebox.utils.dist_adapter as dist

import torch.distributed as dist

import sys
import torch
import os
from random import shuffle
import librosa

mp3_folder = '/srv/audio_mp3s/uploads/5f2b0f6df270d976b43cdafc'
mp3_paths = [f for f in os.listdir(mp3_folder) if f.endswith(".mp3")]
shuffle(mp3_paths)
print(mp3_paths[:5])
filename = 's3_1596681011742_273098292.mp3'
#for filename in mp3_paths[:5]:
mp3_path = os.path.join(mp3_folder, filename)
mp3, _ = librosa.core.load(mp3_path, sr=44100)
print(mp3.shape)


sample_options = {
    "name": "sample_5b",
    "levels": 3,
    "sample_length_in_seconds": 20,
    "total_sample_length_in_seconds": 180,
    "sr": 44100,
    "n_samples": 6,
    "hop_fraction": [0.5, 0.5, 0.125]
}

train_options = {
        "bs": 1,
        "labels": False
}

rank, local_rank, device = setup_dist_from_mpi(port=29500)
print("Device: {}".format(device))

hps = Hyperparams(**sample_options)
hps = setup_hparams("vqvae", dict(sample_length=hps.get('sample_length', 0),
                                  sample_length_in_seconds=hps.get('sample_length_in_seconds', 0),
                                  labels=False, bs=1))


vqvae = make_vqvae(hps, 'cuda:0')
print("sample_length", vqvae.sample_length)
print('multipliers', vqvae.multipliers)
print('x_shape', vqvae.x_shape)
print('downsamples', vqvae.downsamples)
print('hop lengths', vqvae.hop_lengths)
print('z shapes', vqvae.z_shapes)
print('levels', vqvae.levels)

print(len(vqvae.encoders))
# print(vqvae.encoders[0])

print(mp3[:881920].shape)

forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

hps.ngpus = dist.get_world_size()
hps.argv = " ".join(sys.argv)
hps.bs_sample = hps.nworkers = hps.bs

data_processor = DataProcessor(hps)

inputs = torch.tensor(mp3[:881920]).view(1, -1, 1).to(device)
outputs = vqvae(inputs, **forw_kwargs)
