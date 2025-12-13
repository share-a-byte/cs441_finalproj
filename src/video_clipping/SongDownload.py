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
    def __init__(self, capacity):
        self.intervals = [3, 5, 10]
        self.format = "mp3"
        self.ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'postprocessors': [{ 
                'key': 'FFmpegExtractAudio',
                'preferredcodec': self.format,
            }]
        }
        df = pd.read_csv("FINAL.csv")
        self.df = df[['uid', 'type', 'offset']]
        self.capacity = capacity
        self.clip_pool = []

        os.chdir("../..")
        for interval in self.intervals:
            os.makedirs(f"clips/{interval}sec", exist_ok=True)

    def get_clips_length(self):
        min_clips = len(self.df)

        tot_length = 0
        for interval in self.intervals:
            number_intervals = min_clips * (60 / interval)
            tot_length += number_intervals

        return tot_length
    
    def get_clip(self):
        while len(self.clip_pool) < self.capacity and len(self.df) > 0:
            self.download_new_song()
            
        idx = int(np.floor(random.random() * len(self.clip_pool)))
        rand_clip_path, res_type = self.clip_pool.pop(idx) # Popping tuple

        return [rand_clip_path, res_type]

    def download_new_song(self):
        # Step 1. Download_new passed from getitem -> if it is False, we proceed with the clip pool
        sampled = self.df.sample(n=1)
        res = sampled.values[0]
        uid, og_type, offset = res[0], res[1], int(res[2])
        url = f"https://www.youtube.com/watch?v={uid}"

        with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=True)
            norm_filename = ydl.prepare_filename(info_dict)
            song_name = os.path.splitext(norm_filename)[0]
            output_filename = song_name + "." + self.format

            for interval in self.intervals:
                subprocess.run([
                'ffmpeg', '-i', output_filename,
                '-ss', str(offset),  # DOWNLOAD FROM OFFSET
                '-t', "60",          # DO NOT read more than 60 seconds
                '-f', 'segment',
                '-segment_time', str(interval),
                '-c', 'copy',
                f'clips/{interval}sec/{uid}_{offset}_%d.mp3',
                ])

            # Add to clip path the id tuples
            for interval in self.intervals:
                for num in range(60 // interval):
                    self.clip_pool.append((f'clips/{interval}sec/{uid}_{offset}_{num}.mp3', og_type))

            # don't need this file anymore -> comment this out if you still need
            subprocess.call(f'rm "{output_filename}"', shell=True)

        self.df.drop(sampled.index)

if __name__ == "__main__":
    # set local path to be your current directory
        # capacity = 80 -> 80 clips max
            # duration = we pull in this size batch (duration = 60 -> 60 second batch)

    finalized_real = pd.read_csv("FINAL.csv")
    finalized_small = finalized_real[:2]
    finalized_small.to_csv("FINAL.csv") # For testing purposes
    downloader = SongDownloader(capacity=10)
    length = downloader.get_clips_length()

    assert(length == 76.0)

    result1 = downloader.get_clip()
    result2 = downloader.get_clip()

    print(result1, result2)

    os.chdir("src/video_clipping/")
    finalized_real.to_csv("FINAL.csv", index=False)