import yt_dlp
import pandas as pd
import sys
import subprocess
import os

# Constants - change if needed
intervals = [3, 5, 10]
pref_format = "mp3"

# Make sure these folders exist
for interval in intervals:
    os.makedirs(f"clips/{interval}sec", exist_ok=True)


ydl_opts = {
    'format': 'm4a/bestaudio/best',
    'postprocessors': [{ 
        'key': 'FFmpegExtractAudio',
        'preferredcodec': pref_format,
    }]
}

def download_script(url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(url, download=True)
        norm_filename = ydl.prepare_filename(info_dict)
        song_name = os.path.splitext(norm_filename)[0]
        output_filename = song_name + "." + pref_format

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
    df = pd.read_csv("songs.csv")

    if "LINK" not in df:
        sys.exit(0)

    for link in df["LINK"]:
        download_script(link)