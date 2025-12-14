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
        self.format = "wav"
        self.ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }
        path = None
        if "__file__" in globals():
            path = str(Path(__file__).resolve().parent)
        else:
            path = str(Path.cwd().resolve())
        print(globals().get("__file__", "<no __file__>"))
        print(path)
        os.chdir(path)

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
        my_df = None
        if set_type == "train":
            my_df = self.train_df
        elif set_type == "test":
            my_df = self.test_df
        else:
            my_df = self.val_df

        while len(self.clip_pool[set_type]) < self.capacity and len(my_df) > 0:
            self.download_new_song(set_type)
        print('Clip pool filled to capacity!\n')
        idx = int(np.floor(random.random() * len(self.clip_pool[set_type])))
        rand_clip_path, res_type = self.clip_pool[set_type].pop(idx) # Popping tuple

        return [rand_clip_path, (1 if res_type == "AI" else 0)]

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
            try:
              info_dict = ydl.extract_info(url, download=True)
            except:
              time.sleep(1.5)
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
                f'clips/{interval}sec/{uid}_{offset}_%d.wav',
                ], check=False)

            # Add to clip path the id tuples
            for interval in self.intervals:
                clip_dir = f"clips/{interval}sec"
                prefix = f"{uid}_{offset}_"
                for fname in os.listdir(clip_dir):
                    if fname.startswith(prefix):
                        self.clip_pool[set_type].append(
                            (os.path.join(clip_dir, fname), og_type)
                        )
                        
            print('Output file name: {}'.format(output_filename))
            print('CWD: {}'.format(os.getcwd()))
            # don't need this file anymore -> comment this out if you still need
            try:
                os.remove(output_filename)
            except FileNotFoundError:
                pass

        if set_type == "train":
            self.train_df = self.train_df.drop(sampled.index)
        elif set_type == "test":
            self.test_df = self.test_df.drop(sampled.index)
        else:
            self.val_df = self.val_df.drop(sampled.index)