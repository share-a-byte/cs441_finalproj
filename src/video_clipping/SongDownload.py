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
        df = df[['uid', 'type', 'offset']]
        self.df = df

        self.capacity = capacity

        n = len(df)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)

        df_shuffled = df.sample(frac=1, random_state=1).reset_index(drop=True)

        self.train_df = df_shuffled.iloc[:n_train].copy()
        self.val_df = df_shuffled.iloc[n_train:n_train+n_val].copy()
        self.test_df = df_shuffled.iloc[n_train+n_val:].copy()

        self.clip_pool = {"train": [], "test": [], "val": []}

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
    
    def get_clip(self, set_type):
        while len(self.clip_pool[set_type]) < self.capacity and len(self.df) > 0:
            self.download_new_song(set_type)
            
        idx = int(np.floor(random.random() * len(self.clip_pool[set_type])))
        rand_clip_path, res_type = self.clip_pool[set_type].pop(idx) # Popping tuple

        return [rand_clip_path, res_type]

    def download_new_song(self, set_type):
        # Step 1. Download_new passed from getitem -> if it is False, we proceed with the clip pool
        sampled = None

        if set_type == "train":
            sampled = self.train_df.sample(n=1)
        elif set_type == "test":
            sampled = self.test_df.sample(n=1)
        else:
            sampled = self.val_df.sample(n=1)
            
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
                    self.clip_pool[set_type].append((f'clips/{interval}sec/{uid}_{offset}_{num}.mp3', og_type))

            # don't need this file anymore -> comment this out if you still need
            subprocess.call(f'rm "{output_filename}"', shell=True)

        if set_type == "train":
            self.train_df = self.train_df.drop(sampled.index)
        elif set_type == "test":
            self.test_df = self.test_df.drop(sampled.index)
        else:
            self.val_df = self.val_df.drop(sampled.index)

if __name__ == "__main__":
#     # set local path to be your current directory
#         # capacity = 80 -> 80 clips max
#             # duration = we pull in this size batch (duration = 60 -> 60 second batch)

#     # finalized_real = pd.read_csv("FINAL.csv")
#     # finalized_small = finalized_real[:2]
#     # finalized_small.to_csv("FINAL.csv") # For testing purposes
    downloader = SongDownloader(capacity=10)
#     # length = downloader.get_clips_length()

#     print(downloader.train_df.iloc[0])
#     print("------")
#     print(downloader.test_df.iloc[0])
#     print("------")
#     print(downloader.val_df.iloc[0])

#     # assert(length == 76.0)

#     # print(result1, result2)
    
    downloader.get_clip("train")
    print(downloader.clip_pool)

#     # os.chdir("src/video_clipping/")
#     # finalized_real.to_csv("FINAL.csv", index=False)