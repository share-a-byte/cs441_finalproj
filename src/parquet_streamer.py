import pyarrow as pa
import pyarrow.parquet as pq
import os
from data_processing import AudioProcessing as AP
from video_clipping import SongDownload 
from SongDownload import SongDownloader

from pathlib import Path
import kagglehub
import os

def stream_to_parquet(downloader, out_path, chunk_size=128):
    rows = []
    writer = None

    noise_path = Path(kagglehub.dataset_download("moazabdeljalil/back-ground-noise")).absolute()
    noise_list = []

    for root, dirs, files in os.walk(noise_path):
        for file in files:
            full_path = os.path.join(root, file)
            noise_list.append(os.path.abspath(full_path))

    while True:
        try:
            clip_path, label = downloader.get_next_clip()

            audio = AP.Utils.get_audio_and_rechannel(clip_path, 2)
            audio = AP.Utils.resample(audio, 44100)
            audio = AP.Utils.time_shift(audio, 0.4)
            audio = AP.Utils.add_noise(audio, noise_list)

            spec = AP.Utils.spectrogram(audio)
            spec = AP.Utils.augment_spectrogram(spec, 44100)

            rows.append({
                "spec": spec.numpy().astype("float32").flatten(),
                "label": label
            })

            try:
                os.remove(clip_path)
            except FileNotFoundError:
                pass

            if len(rows) >= chunk_size:
                table = pa.Table.from_pylist(rows)
                if writer is None:
                    writer = pq.ParquetWriter(out_path, table.schema)
                writer.write_table(table)
                rows.clear()

        except StopIteration:
            break

    if rows:
        table = pa.Table.from_pylist(rows)
        writer.write_table(table)

    if writer:
        writer.close()

if __name__ == "__main__":
    downloader = SongDownloader(50)
    stream_to_parquet(downloader, Path(__file__).resolve().parent + "/parq_data/")