import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import shutil
import os

import data_processing.AudioProcessing as AP
from src.video_clipping import SongDownload

df = pd.read_csv("data/songs.csv")

class SoundDataset(Dataset):
    def __init__(self, local_path, duration=1000, sr=44100, clip_pool_capacity=80):
        self.duration = duration
        self.sr = sr
        self.channel = 2
        self.shift_pct = 0.4
        self.downloader = SongDownload.SongDownloader(local_path=local_path, capacity=clip_pool_capacity)
        self.local_path = local_path

    def __len__(self):
        return len(self.df)   

    def __getitem__(self):
        clip = self.downloader.get_clip()
        aud = AP.Utils.get_audio_and_rechannel(audio_file, self.n_channels)
    reaud = AP.Utils.resample(aud, self.sr)
    shift_aud = AP.Utils.time_shift(reaud, self.shift_pct)
    sgram = AP.Utils.spectrogram(shift_aud)
    aug_sgram = AP.Utils.augment_spectrogram(sgram, max_mask=0.1, n_fmask=2, n_tmask=2)
        os.remove(os.path.join(self.local_path, clip))
        # eero's resampling code
        shutil.rmtree(self.temp_path)
