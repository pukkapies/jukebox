from jukebox.hparams import setup_hparams, Hyperparams
from jukebox.make_models import make_vqvae
from jukebox.utils.dist_utils import setup_dist_from_mpi
from jukebox.utils.audio_utils import audio_preprocess, spec

import sys
import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import shutil
import torch
import os
from random import shuffle
import librosa


def load_json(path):
    with open(path, 'r') as json_file:
        j = json.load(json_file)
    return j


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


def save_spec_plot(spec, path, title=None):
    fig = plt.figure(figsize=(7, 7))
    plt.imshow(spec)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    if title:
        plt.title(title)
    plt.savefig(path, bbox_inches='tight')
    fig.close()


# mp3_folder = '/srv/audio_mp3s/uploads/5f2b0f6df270d976b43cdafc'
audio_mp3s_folder = 'srv/audio_mp3s'
json_path = '/home/kevin/feedforward/mp3s_for_jukebox_test.json'
mp3_dict = load_json(json_path)
output_folder = '/home/kevin/pukkapies_github/jukebox/tests'
# mp3_paths = [f for f in os.listdir(mp3_folder) if f.endswith(".mp3")]
# shuffle(mp3_paths)
# print(mp3_paths[:5])
# filename = 's3_1596681011742_273098292.mp3'
#for filename in mp3_paths[:5]:
# mp3_path = os.path.join(mp3_folder, filename)
csv = {"client": [], "media_id": [], "external_id": [], "s3_key": [], 'recons_loss_l3': [], 'spectral_loss_l3':[],
       'multispectral_loss_l3': [], 'recons_loss_l2': [], 'spectral_loss_l2': [], 'multispectral_loss_l2': [],
       'recons_loss_l1': [], 'spectral_loss_l1': [], 'multispectral_loss_l1': [], 'recons_loss': [],
       'spectral_loss': [], 'multispectral_loss': [], 'spectral_convergence': [], 'l2_loss': [], 'l1_loss': [],
       'linf_loss': [], 'commit_loss': []
       }

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

forw_kwargs = dict(loss_fn=hps.loss_fn, hps=hps)

# hps.ngpus = dist.get_world_size()
hps.argv = " ".join(sys.argv)
hps.bs_sample = hps.nworkers = hps.bs = 1


for client_name in mp3_dict:
    if not os.path.exists(os.path.join(output_folder, client_name)):
        os.makedirs(os.path.join(output_folder, client_name))

    for mp3_metadata in mp3_dict:  # 'external_id', 'media_id', 'num_samples', 's3_key'
        s3_key = mp3_metadata['s3_key']
        filename = s3_key.split('/')[-1]
        mp3_path = os.path.join(audio_mp3s_folder, s3_key)
        mp3, _ = librosa.core.load(mp3_path, sr=44100)
        shutil.copyfile(mp3_path, os.path.join(output_folder, client_name, 'audio', mp3_path.split('/')[-1]))
        mp3_spec = spec(mp3, hps)
        save_spec_plot(mp3_spec, os.path.join(output_folder, client_name, 'spec', filename.split('.')[0], '.png'),
                       title=filename.split('.')[0])

        hps.bandwidth = get_bandwidth(mp3, hps)
        inputs = torch.tensor(mp3[:881920]).view(1, -1, 1).to(device)
        inputs = audio_preprocess(inputs, hps)
        x_out, loss, _metrics = vqvae(inputs, **forw_kwargs)

        # print("Loss: {}".format(loss))
        # print("Metrics:", _metrics)
        x_out_np = x_out.cpu().squeeze().copy().numpy()
        librosa.output.write_wav("{}/{}_recon.wav".format(os.path.join(output_folder, client_name, 'audio'),
                                                          filename.split('.')[0]),
                                 x_out_np, sr=44100)
        x_out_spec = spec(x_out_np, hps)
        save_spec_plot(x_out_spec, os.path.join(output_folder, client_name, 'spec', filename.split('.')[0], '_recon.png'),
                       title=filename.split('.')[0] + " reconstruction")

        csv['client'].append(client_name)
        csv['media_id'].append(mp3_metadata['media_id'])
        csv['s3_key'].append(mp3_metadata['s3_key'])
        for k, v in _metrics:
            csv[k].append(float(v.squeeze().numpy()))

        pd.DataFrame(csv).to_csv(os.path.join(output_folder, 'metrics.csv'))

