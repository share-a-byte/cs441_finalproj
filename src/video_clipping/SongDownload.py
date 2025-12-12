import yt_dlp
import pandas as pd
import sys
import subprocess
import os
from pathlib import Path
import random
import numpy as np

# Make sure these folders exist
class SongDownloader:
    def __init__(self, local_path, capacity):
        self.intervals = [3, 5, 10]
        self.format = "mp3"
        self.ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'postprocessors': [{ 
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.format,
            }]
        }
        df = pd.read_csv("finalized.csv")
        self.df = pd.DataFrame(['LINK', 'ID', 'COMPLETED', 'POSITION'])
        self.local_path = local_path
        self.capacity = capacity
        self.clip_pool = []
    
    def get_clip(self):
        while(len(self.clip_pool) < self.capacity):
            self.download_script()
        idx = np.floor(random.random() * len(self.clip_pool))
        rand_clip_path = self.clip_pool.pop(idx)
        return rand_clip_path
        

    def download_script(self):
        # Step 1. Download_new passed from getitem -> if it is False, we proceed with the clip pool
        # Step 2. 
        for interval in self.intervals:
            os.makedirs(f"clips/{interval}sec", exist_ok=True)

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            norm_filename = ydl.prepare_filename(info_dict)
            song_name = os.path.splitext(norm_filename)[0]
            output_filename = song_name + "." + self.format

            for interval in intervals:
                subprocess.run([
                'ffmpeg', '-i', output_filename,
                '-f', 'segment',
                '-segment_time', str(interval),
                '-c', 'copy',
                f'clips/{interval}sec/{song_name}_%03d.mp3',
                ])

            # don't need this file anymore -> comment this out if you still need
            subprocess.call(f'rm "{output_filename}"', shell=True)

if __name__ == "__main__":
    base_path = Path(__file__).parent
    os.chdir(base_path)

    df = pd.read_csv("songs.csv")

    if "LINK" not in df:
        sys.exit(0)
    for link in df["LINK"]: