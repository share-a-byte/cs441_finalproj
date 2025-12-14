import numpy as np
import random
import torch
import torchaudio
from torchaudio import transforms
import torchaudio.functional as F

class Utils():
	@staticmethod
	def resample(audio, new_sr):
		sig, sr = audio
		if(sr == new_sr):
			return audio
		n_channels = sig.shape[0]
		re_sig = torchaudio.transforms.Resample(sr, new_sr)(sig[:1, :])
		if(n_channels > 1):
			re_sec = transforms.Resample(sr, new_sr)(sig[1:, :])
			re_sig = torch.cat([re_sig, re_sec])
		return (re_sig, new_sr)

	#from Towards Data Science: audio classification II
	@staticmethod
	def time_shift(audio, max_shift):
		sig, sr = audio
		slen = sig.shape[1]
		shift = int(random.random() * max_shift * slen)
		print('Shifting by {}'.format(shift))
		new_sig = sig.roll(shift)
		return (new_sig, sr)

	@staticmethod
	def add_noise(audio, noise_list, min_snr=0, max_snr = 5):
		audio_data, sr = audio
		random_noise_file = random.choice(noise_list)

		noise_audio = Utils.get_audio_and_rechannel(random_noise_file, 2)
		noise, _ = Utils.resample(noise_audio, new_sr=sr)

		audio_length = audio_data.shape[-1]
		noise_length = noise.shape[-1]
		print(noise_length, audio_length)
		if noise_length > audio_length:
			offset = random.randint(0, noise_length-audio_length)
			noise = noise[..., offset:offset+audio_length]
		elif noise_length < audio_length:
			repeats = torch.ceil(torch.tensor(audio_length / noise_length)).long().item()
			tiled_noise = torch.tile(noise, (1, repeats))
			noise = tiled_noise[..., :audio_length]
		print('Noise shape: {}, Audio shape: {}'.format(noise.shape, audio_data.shape))
		snr_db = random.randint(min_snr, max_snr)
		snr = np.exp(snr_db / 10)
		audio_power = audio_data.norm(p=2)
		noise_power = noise.norm(p=2)
		scale = snr * noise_power / audio_power
		return ((scale * audio_data + noise ) / 2, sr)

	@staticmethod
	def spectrogram(audio, n_mels=64, n_fft=2048, hop_len=None):
		sig, sr = audio
		spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)
		spec = F.amplitude_to_DB(spec, multiplier=10.0, amin=1e-10, db_multiplier=0.0, top_db=80.0)
		return spec

	@staticmethod
	def augment_spectrogram(spec, sr, n_fft=2048, target_duration=10, max_mask=0.1, n_fmask=1, n_tmask=1):
		_, n_mels, n_steps = spec.shape
		mask_val = spec.mean()
		aug_spec = spec
		freq_mask_val = n_mels * max_mask
		time_mask_val = n_steps * max_mask	
		for _ in range(n_fmask):
			aug_spec = transforms.FrequencyMasking(freq_mask_param=freq_mask_val)(aug_spec, mask_val)
		for _ in range(n_tmask):
			aug_spec = transforms.TimeMasking(time_mask_param=time_mask_val)(aug_spec, mask_val)
		target_frames = int(np.ceil((target_duration * sr) / n_fft)) #assumes no hop length
		current_frames = aug_spec.shape[1]
		pad_frames = target_frames - current_frames
		pad_aug_spec = torch.nn.functional.pad(aug_spec, (pad_frames//2, pad_frames//2), mode="constant", value=mask_val)
		
		return pad_aug_spec

	@staticmethod
	def get_audio_and_rechannel(file_path, nchannels):
		sig, sr = torchaudio.load(file_path)
		if (sig.shape[0] == nchannels):
			return ((sig, sr))
		if (nchannels == 1):
			resig = sig[:1, :]
		else:
			resig = torch.cat([sig, sig])
		return (resig, sr)
			
