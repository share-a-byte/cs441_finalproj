import yt_dlp
import pandas as pd
import subprocess
import os
from pathlib import Path
import time

class SongDownloader:
    def __init__(self, capacity, max_retries=3):
        self.intervals = [3, 5, 10]
        self.format = "wav"
        self.ydl_opts = {
            'format': 'm4a/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }]
        }

        path = str(Path(__file__).resolve().parent) if "__file__" in globals() else str(Path.cwd())
        os.chdir(path)

        df = pd.read_csv("FINAL.csv")
        df = df[['uid', 'type', 'offset']]
        self.df = df.sample(frac=1, random_state=1).reset_index(drop=True)

        self.cursor = 0
        self.capacity = capacity
        self.clip_pool = []

        self.retry_counts = {}
        self.max_retries = max_retries

        for interval in self.intervals:
            os.makedirs(f"clips/{interval}sec", exist_ok=True)

    def get_next_clip(self):
        while len(self.clip_pool) < self.capacity:
            self.download_next_song()

        if len(self.clip_pool) == 0:
            raise StopIteration # no more valid clips

        return self.clip_pool.pop()

    def download_next_song(self):
        while self.cursor < len(self.df):
            row = self.df.iloc[self.cursor]
            self.cursor += 1

            uid, og_type, offset = row["uid"], row["type"], int(row["offset"])
            print("Downloading song ", self.cursor, " retry number ", self.retry_counts.get(uid, 0))
            
            label = 1 if og_type == "AI" else 0
            url = f"https://www.youtube.com/watch?v={uid}"

            attempts = self.retry_counts.get(uid, 0)
            if attempts >= self.max_retries:
                continue

            try:
                with yt_dlp.YoutubeDL(self.ydl_opts) as ydl:
                    info_dict = ydl.extract_info(url, download=True)
            except Exception:
                self.retry_counts[uid] = attempts + 1
                time.sleep(0.5)
                continue

            self.retry_counts.pop(uid, None)

            norm_filename = ydl.prepare_filename(info_dict)
            song_name = os.path.splitext(norm_filename)[0]
            output_filename = song_name + "." + self.format

            for interval in self.intervals:
                subprocess.run([
                    'ffmpeg', '-i', output_filename,
                    '-ss', str(offset),
                    '-t', "60",
                    '-f', 'segment',
                    '-segment_time', str(interval),
                    '-c', 'copy',
                    f'clips/{interval}sec/{uid}_{offset}_%d.wav',
                ], check=False)

            try:
                os.remove(output_filename)
            except FileNotFoundError:
                pass

            for interval in self.intervals:
                clip_dir = f"clips/{interval}sec"
                prefix = f"{uid}_{offset}_"
                for fname in os.listdir(clip_dir):
                    if fname.startswith(prefix):
                        self.clip_pool.append(
                            (os.path.join(clip_dir, fname), label)
                        )

            if len(self.clip_pool) > 0:
                return

        raise StopIteration
