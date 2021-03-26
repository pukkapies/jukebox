from jukebox.hparams import setup_hparams, Hyperparams
from jukebox.make_models import make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.data.data_processor import DataProcessor
import jukebox.utils.dist_adapter as dist
from jukebox.utils.audio_utils import audio_preprocess

import torch.distributed as dist

import sys
import numpy as np
import torch
import os
from random import shuffle
import librosa

def get_bandwidth(mp3, hps):
    stft = librosa.core.stft(mp3, hps.n_fft, hop_length=hps.hop_length, win_length=hps.window_size)
    spec = np.absolute(stft)
    spec_norm_total = np.linalg.norm(spec)
    spec_nelem = 1
    n_seen = int(np.prod(mp3.shape))
    l1 = np.sum(np.abs(mp3))
    total = np.sum(mp3)
    total_sq = np.sum(mp3 ** 2)
    mean = total / n_seen
    bandwidth = dict(l2=total_sq / n_seen - mean ** 2,
                     l1=l1 / n_seen,
                     spec=spec_norm_total / spec_nelem)
    return bandwidth

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

# hps.ngpus = dist.get_world_size()
hps.argv = " ".join(sys.argv)
hps.bs_sample = hps.nworkers = hps.bs = 1

hps.bandwidth = get_bandwidth(mp3, hps)
inputs = torch.tensor(mp3[:881920]).view(1, -1, 1).to(device)
inputs = audio_preprocess(inputs, hps)
x_out, loss, _metrics = vqvae(inputs, **forw_kwargs)

print("Loss: {}".format(loss))
print("Metrics:", _metrics)
