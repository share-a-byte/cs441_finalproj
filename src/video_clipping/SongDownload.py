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
    def __init__(self, local_path, capacity, duration):
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
        self.df = df[['id', 'duration', 'type']]
        self.df['offset'] = 0           # all data starts off with an offset of 0
        self.local_path = local_path
        self.capacity = capacity
        self.clip_pool = []
        self.duration = duration # IN SECONDS

        for interval in self.intervals:
            os.makedirs(f"clips/{interval}sec", exist_ok=True)

    def get_clips_length(self):
        number_minutes = 0

        for row in self.df.iterrows():
            number_minutes += (row["duration"] // self.duration)

        for interval in self.intervals:
            number_intervals = number_minutes * (self.duration / interval)

        return number_intervals
    
    def get_clip(self):
        while len(self.clip_pool) < self.capacity and len(self.df) > 0:
            self.download_new_song()
            
        idx = np.floor(random.random() * len(self.clip_pool))
        rand_clip_path, res_type = self.clip_pool.pop(idx) # Popping tuple

        return [rand_clip_path, res_type]

    def download_new_song(self):
        # Step 1. Download_new passed from getitem -> if it is False, we proceed with the clip pool
        id, duration, og_type, offset = self.df.sample(n=1)
        url = f"https://www.youtube.com/watch?v={id}"

        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            norm_filename = ydl.prepare_filename(info_dict)
            song_name = os.path.splitext(norm_filename)[0]
            output_filename = song_name + "." + self.format

            for interval in self.intervals:
                subprocess.run([
                'ffmpeg', '-i', output_filename,
                '-ss ', f'{offset}' # DOWNLOAD FROM OFFSET
                '-f', 'segment',
                '-segment_time', str(interval),
                '-c', 'copy',
                f'clips/{interval}sec/{offset}_%d.mp3',
                ])

            # Add to clip path the id tuples
            for interval in self.intervals:
                for num in (1, self.duration // interval + 1):
                    self.clips.append((f'clips/{interval}sec/{offset}_{num}.mp3', og_type))

            # don't need this file anymore -> comment this out if you still need
            subprocess.call(f'rm "{output_filename}"', shell=True)

        offset += 60
        index = self.df[self.df['id'] == id].index

        if (offset + 60) > duration: # Can't read another 60 second chunk
            self.df.drop(index)
        else:
            self.df[index] = [id, duration, og_type, offset + 60]

# if __name__ == "__main__":
#     base_path = Path(__file__).parent
#     os.chdir(base_path)

#     df = pd.read_csv("songs.csv")

#     if "LINK" not in df:
#         sys.exit(0)

#     for link in df["LINK"]: