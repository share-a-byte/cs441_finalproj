import numpy as np
import random
import torch
import torchaudio
from torchaudio import transforms

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
	return ((sig.roll(shift), sr))

def add_noise(audio, noise_list, min_snr=0.01, max_snr = 0.2):
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
	return (((scale * audio_data + noise ) / 2, sr))

def spectrogram(audio, max_duration, n_mels=64, n_fft=2048, hop_len=None):
	sig, sr = audio
	nsamples = sig.shape[-1]
	max_samples = max_duration * sr
	padding = max_samples - nsamples
	pad_sig = torch.nn.functional.pad(sig, (padding/2, padding/2), "constant", 0)
	spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(pad_sig)
	return spec

def augment_spectrogram(spec, max_mask=0.1, n_fmask=1, n_tmask=1):
	_, n_mels, n_steps = spec.shape
	mask_val = spec.mean()
	aug_spec = spec
	freq_mask_val = n_mels * max_mask
	time_mask_val = n_steps * max_mask	
	for _ in range(n_fmask):
		aug_spec = transforms.FrequencyMasking(freq_mask_param=freq_mask_val)(aug_spec, mask_val)
	for _ in range(n_tmask):
		aug_spec = transforms.TimeMasking(time_mask_param=time_mask_val)(aug_spec, mask_val)
	return aug_spec

def get_audio_and_resample(file_path, sample_rate):
	sig, sr = torchaudio.load(file_path)
	re_sig = sig
	audio = (re_sig, sr)
	if(sig.shape[0] != 2):
		re_sig = torch.cat([sig, sig])
		audio = resample((re_sig, sr), sample_rate)
	return audio
		
