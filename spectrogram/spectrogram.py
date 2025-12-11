import numpy as np
import kagglehub
import random
import torch
import torchaudio
from torchaudio import transforms, functional
import matplotlib.pyplot as plt
import pathlib
from torchaudio.utils import download_asset

import os
n_fft = 2048 
hoplen = 512
nbatches = 3
min_snr = 0.01
max_snr = 0.2


def prepare_training_data(clip_path):
	global nbatches
	x_train = np.empty(3, dtype=object)
	y_train = np.empty(3, dtype=object)
	for i in range(nbatches):
		x_train[i] = []
		y_train[i] = []
	for dir in os.listdir(clip_path):
		dir_path = os.path.join(clip_path, dir)
		if(os.path.isdir(dir_path)):
			for subdir in os.listdir(dir_path):
				sub_dir_path = os.path.join(dir_path, subdir)
				if(os.path.isdir(sub_dir_path)):
					for file in os.listdir(sub_dir_path):
						file_path = os.path.join(sub_dir_path, file)
						is_ai = (subdir == "AI")
						if(dir == "3sec"):
							x_train[0].append(file_path)
							y_train[0].append(is_ai)
						elif(dir == "5sec"):
							x_train[1].append(file_path)
							y_train[1].append(is_ai)
						else:
							x_train[2].append(file_path)
							y_train[2].append(is_ai)
	return x_train, y_train

def resample(audio, new_sr):
	sig, sr = audio
	if(sr == new_sr):
		return audio
	n_channels = sig.shape[0]
	re_sig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
	if(n_channels > 1):
		re_sec = transforms.Resample(sr, new_sr)(sig[1:, :])
		re_sig = torch.cat([re_sig, re_sec])
	return ((re_sig, new_sr))

#from Towards Data Science: audio classification II
def time_shift(audio, max_shift):
	sig, sr = audio
	slen = sig.shape[0]
	shift = int(random.random * max_shift * slen)
	return (sig.roll(shift), sr)

def add_noise(audio, noise_list):
	global min_snr, max_snr
	audio_data, sr = audio
	random_noise_file = random.choice(noise_list)
	effects = [
		['remix', '2'], # convert to mono
        ['rate', str(sr)], # resample
    ]
	noise, _ = torchaudio.sox_effects.apply_effects_file(random_noise_file, effects, normalize=True)
	audio_length = audio_data.shape[-1]
	noise_length = noise.shape[-1]
	if noise_length > audio_length:
		offset = random.randint(0, noise_length-audio_length)
		noise = noise[..., offset:offset+audio_length]
	elif noise_length < audio_length:
		noise = torch.cat([noise, torch.zeros((noise.shape[0], audio_length-noise_length))], dim=-1)
	snr_db = random.randint(min_snr, max_snr)
	snr = np.exp(snr_db / 10)
	audio_power = audio_data.norm(p=2)
	noise_power = noise.norm(p=2)
	scale = snr * noise_power / audio_power
	return (scale * audio_data + noise ) / 2


def process_audio(audiofiles, duration, sample_rate):
	for x in audiofiles:
		sig, sr = torchaudio.load(x)
		re_sig = sig
		if(sig.shape[0] != 2):
			re_sig = torch.cat([sig, sig])
		audio = resample((re_sig, sr), sample_rate)
		
if __name__ == "__main__":
	noise_path = kagglehub.dataset_download("moazabdeljalil/back-ground-noise")
	x_t, y_t = prepare_training_data('../video_clipping/clips')
	for i in range(len(x_t[0])):
		label = "is not AI-Generated"
		if(y_t[0][i]): 
			label = "is AI-Generated"
		print('File {} {}'.format(x_t[0][i], label))