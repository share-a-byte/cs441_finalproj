import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split
import shutil

from data_processing import AudioProcessing
from src.video_clipping import SongDownload

df = pd.read_csv("data/songs.csv")

class SoundDataset(Dataset):
    def __init__(self, df):
        self.df = df
        self.temp_path = "video_clipping/clips/"
        self.duration = 1000
        self.sr = 44100
        self.channel = 2
        self.shift_pct = 0.4

    def __len__(self):
        return len(self.df)   

    def __getitem__(self, idx):
        link = self.df.loc[idx, 'LINK']
        SongDownload.download_script(link)
        # eero's resampling code
        shutil.rmtree(self.temp_path)

class AudioCNN(nn.Module):
    def __init__(self, num_classes=10, dropout = 0.5):
        super(AudioCNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
			nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout(dropout),
            nn.Linear(256 * 4 * 4, 256),
            nn.Linear(256, num_classes)
		)
    def forward(self, x):
        return self.classifier(x)
       
