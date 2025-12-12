import data_processing.AudioProcessing as AP
import pandas as pd
from src.video_clipping import SongDownload
import shutil

class SongDataset(Dataset):
  def __init__(self, data_path):
    df = pd.read_csv(data_path)
    self.index = 0

    files = set()

    self.temp_path = "video_clipping/clips/"
    
    self.duration = 4000
    self.sr = 44100
    self.n_channels = 2
    self.shift_pct = 0.4

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    link = self.df.loc[idx, 'LINK']
    SongDownload.download_script(link)
    
    audio_file = self.temp_path + self.df.loc[idx, 'relative_path']
    class_id = self.df.loc[idx, 'classID']
    aud = AP.Utils.get_audio_and_rechannel(audio_file, self.n_channels)
    reaud = AP.Utils.resample(aud, self.sr)
    shift_aud = AP.Utils.time_shift(reaud, self.shift_pct)
    sgram = AP.Utils.spectrogram(shift_aud)
    aug_sgram = AP.Utils.augment_spectrogram(sgram, max_mask=0.1, n_fmask=2, n_tmask=2)

    shutil.rmtree(self.temp_path)
    return aug_sgram, class_id