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


def save_as_json(d, path, sort_keys=True):
    with open(path, 'w+') as json_file:
        json.dump(d, json_file, sort_keys=sort_keys, indent=4)


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
    if type(spec) == np.ndarray:
        fig = plt.figure(figsize=(25, 5))
        plt.imshow(librosa.core.power_to_db(spec[::-1, :]))
        plt.xlabel("Time")
        plt.ylabel("Frequency")
        if title:
            plt.title(title)
        plt.colorbar()
    elif type(spec) == list:
        assert len(spec) == 4
        for s in spec:
            assert type(s) == np.ndarray
        fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(30, 10))
        fig.tight_layout()

        ax1 = axs[0]
        im = ax1.imshow(librosa.core.power_to_db(spec[0][::-1, :]), vmin=-60, vmax=20)
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Frequency")
        if title:
            ax1.set_title(title + " original", fontsize=10)

        for i in range(1, 4):
            im = axs[i].imshow(librosa.core.power_to_db(spec[i][::-1, :]), vmin=-60, vmax=20)
            axs[i].set_xlabel("Time")
            axs[i].set_ylabel("Frequency")
            if title:
                axs[i].set_title(title + " reconstruction level {}".format(i-1), fontsize=10)

        fig.subplots_adjust(right=0.83)
        cbar_ax = fig.add_axes([0.7, 0.1, 0.01, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        fig.subplots_adjust(hspace=0.6)
    else:
        raise NotImplementedError("spec arg must be array or list of 2 arrays")

    plt.savefig(path, bbox_inches='tight')
    plt.close()



# mp3_folder = '/srv/audio_mp3s/uploads/5f2b0f6df270d976b43cdafc'
audio_mp3s_folder = '/s3/figaro-api/'
output_folder = '/home/ubuntu/jukebox/tests'

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
# print(hps)

vqvae = make_vqvae(hps, 'cuda:0')


def compute_metrics(vqvae, hps, output_folder):
    json_path = '/home/kevin/feedforward/mp3s_for_jukebox_test.json'
    mp3_dict = load_json(json_path)
    csv = {"client": [], "media_id": [], "external_id": [], "s3_key": [], 'recons_loss_l3': [], 'spectral_loss_l3':[],
           'multispectral_loss_l3': [], 'recons_loss_l2': [], 'spectral_loss_l2': [], 'multispectral_loss_l2': [],
           'recons_loss_l1': [], 'spectral_loss_l1': [], 'multispectral_loss_l1': [], 'recons_loss': [],
           'spectral_loss': [], 'multispectral_loss': [], 'spectral_convergence': [], 'l2_loss': [], 'l1_loss': [],
           'linf_loss': [], 'commit_loss': []
           }

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
        if not os.path.exists(os.path.join(output_folder, client_name, 'audio')):
            os.makedirs(os.path.join(output_folder, client_name, 'audio'))
        if not os.path.exists(os.path.join(output_folder, client_name, 'spec')):
            os.makedirs(os.path.join(output_folder, client_name, 'spec'))

        for mp3_metadata in mp3_dict[client_name]:  # 'external_id', 'media_id', 'num_samples', 's3_key'
            print(mp3_metadata)
            s3_key = mp3_metadata['s3_key']
            filename = s3_key.split('/')[-1]
            mp3_path = os.path.join(audio_mp3s_folder, s3_key)
            mp3, _ = librosa.core.load(mp3_path, sr=44100)
            librosa.output.write_wav("{}/{}.wav".format(os.path.join(output_folder, client_name, 'audio'),
                                                        filename.split('.')[0]),
                                     mp3[:881920], sr=44100)

            hps.bandwidth = get_bandwidth(mp3, hps)
            inputs = torch.tensor(mp3[:881920]).view(1, -1, 1).to(device)

            mp3_spec = spec(inputs.squeeze().cpu(), hps).numpy()
            # save_spec_plot(mp3_spec, os.path.join(output_folder, client_name, 'spec', filename.split('.')[0] + '.png'),
            #                title=filename.split('.')[0])

            inputs = audio_preprocess(inputs, hps)
            x_outs, loss, _metrics = vqvae(inputs, **forw_kwargs, return_all_x_outs=True)  # x_outs with top level first

            # print("Loss: {}".format(loss))
            # print("Metrics:", _metrics)

            out_specs = []
            for i, x_out in enumerate(reversed(x_outs)):  # level 0 (bottom) first
                x_out_np = x_out.cpu().squeeze().numpy()
                librosa.output.write_wav("{}/{}_recon{}.wav".format(os.path.join(output_folder, client_name, 'audio'),
                                                                  filename.split('.')[0], i),
                                         x_out_np, sr=44100)
                x_out_spec = spec(x_out.squeeze().cpu(), hps).numpy()
                out_specs.append(x_out_spec)

            save_spec_plot([mp3_spec] + out_specs, os.path.join(output_folder, client_name, 'spec', filename.split('.')[0] + '.png'),
                           title=filename.split('.')[0])

            csv['client'].append(client_name)
            csv['media_id'].append(mp3_metadata['media_id'])
            csv['external_id'].append(mp3_metadata['external_id'])
            csv['s3_key'].append(mp3_metadata['s3_key'])
            for k, v in _metrics.items():
                csv[k].append(float(v.squeeze().cpu().numpy()))

            pd.DataFrame(csv).to_csv(os.path.join(output_folder, 'metrics.csv'))


def compute_codes(vqvae, hps, output_folder, json_path):
    # json_path = '/home/ubuntu/jukebox/jukebox/mp3s_for_jukebox_encodings.json'
    mp3_dict = load_json(json_path)

    hps.argv = " ".join(sys.argv)
    hps.bs_sample = hps.nworkers = hps.bs = 1

    for client_name in mp3_dict:
        level_0_codes, level_1_codes, level_2_codes = [], [], []
        successful_mp3s = []
        unsuccessful_mp3s = []

        if not os.path.exists(os.path.join(output_folder, client_name)):
            os.makedirs(os.path.join(output_folder, client_name))

        for mp3_metadata in mp3_dict[client_name]:  # 'external_id', 'media_id', 'num_samples', 's3_key'
            s3_key = mp3_metadata['s3_key']
            filename = s3_key.split('/')[-1]
            mp3_path = os.path.join(audio_mp3s_folder, s3_key)

            try:
                mp3, _ = librosa.core.load(mp3_path, sr=44100)
                hps.bandwidth = get_bandwidth(mp3, hps)

                max_length_per_pass = int(5e6)
                num_chunks = int(np.ceil(mp3.size / max_length_per_pass))

                full_codes0, full_codes1, full_codes2 = [], [], []
                for i in range(num_chunks):
                    inputs = torch.tensor(mp3[i*max_length_per_pass: (i+1)*max_length_per_pass]).view(1, -1, 1).to(device)

                    # inputs = audio_preprocess(inputs, hps)  # This doesn't do anything anyway
                    codes0, codes1, codes2 = vqvae.encode(inputs)
                    full_codes0.extend(codes0.detach().cpu().numpy().tolist())
                    full_codes1.extend(codes1.detach().cpu().numpy().tolist())
                    full_codes2.extend(codes2.detach().cpu().numpy().tolist())

                level_0_codes.append(full_codes0)
                level_1_codes.append(full_codes1)
                level_2_codes.append(full_codes2)
                mp3_metadata['num_samples'] = int(mp3.size)
                successful_mp3s.append(mp3_metadata)
            except:
                unsuccessful_mp3s.append(mp3_path)
                continue

        save_as_json(level_0_codes, os.path.join(output_folder, client_name, 'level_0_codes.json'))
        save_as_json(level_1_codes, os.path.join(output_folder, client_name, 'level_1_codes.json'))
        save_as_json(level_2_codes, os.path.join(output_folder, client_name, 'level_2_codes.json'))

        save_as_json(successful_mp3s, os.path.join(output_folder, client_name, 'successful_mp3s.json'))
        save_as_json(unsuccessful_mp3s, os.path.join(output_folder, client_name, 'unsuccessful_mp3s.json'))


# compute_metrics(vqvae, hps, output_folder)
# compute_codes(vqvae, hps, output_folder)

if __name__ == '__main__':
    json_path = sys.argv[1]
    compute_codes(vqvae, hps, output_folder, json_path)
