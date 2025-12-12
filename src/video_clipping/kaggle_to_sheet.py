from yt_dlp import YoutubeDL
import pandas as pd
from pathlib import Path
import os
import time

if __name__ == "__main__":
    original_data = pd.read_csv("songs.csv")
    cols_to_keep = ['Title', 'LINK']
    original_data = original_data[cols_to_keep]

    base_path = Path(__file__).parent   # where this script is
    os.chdir(base_path)
    
    kaggle_dataset = pd.read_csv("../ReggaeHits.csv")

    with YoutubeDL({"format": "bestaudio/best", "quiet": True}) as ydl:
        for index, row in kaggle_dataset.iterrows():
            artist = row["Artist"]
            song_name = row["Track"]

            query = f"ytsearch:{artist} {song_name}"

            try:
                info = ydl.extract_info(query, download=False)
                vid_id = info["entries"][0]["id"]
                title = info["entries"][0]["title"]
                yt_url = f"https://www.youtube.com/watch?v={vid_id}"
                
                original_data.loc[len(original_data)] = [title, yt_url]
                time.sleep(0.5)
            except:
                continue

    original_data.to_csv("finalized.csv")