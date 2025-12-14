import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import subprocess
import os

import data_processing.AudioProcessing as AP
import video_clipping.SongDownload as SongDownload

def spec_gram_collate(batch):
    spectrograms = [item[0] for item in batch]
    ids = [item[1] for item in batch]
    batched_spectrograms = torch.stack(spectrograms, dim=0)
    return batched_spectrograms, ids

class SoundDataset(Dataset):
    def __init__(self, noise_list, set_type="train", sr=44100, clip_pool_capacity=80):
        self.sr = sr
        self.n_channels = 2
        self.shift_pct = 0.4
        self.downloader = SongDownload.SongDownloader(capacity=clip_pool_capacity)
        self.noise_list = noise_list
        self.set_type = set_type

    def __len__(self):
        return int(self.downloader.get_clips_length())

    def __getitem__(self, index):
        clip, id = self.downloader.get_clip(self.set_type)
        aud = AP.Utils.get_audio_and_rechannel(clip, self.n_channels)
        reaud = AP.Utils.resample(aud, self.sr)
        shift_aud = AP.Utils.time_shift(reaud, self.shift_pct)
        noised_aud = AP.Utils.add_noise(shift_aud, self.noise_list)
        sgram = AP.Utils.spectrogram(noised_aud)
        aug_sgram = AP.Utils.augment_spectrogram(sgram, self.sr, max_mask=0.1, n_fmask=2, n_tmask=2)
        subprocess.call(f'del "{clip}"', shell=True)
        return aug_sgram, id