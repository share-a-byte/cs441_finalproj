from video_clipping import SongDownload
from SongDownload import SongDownloader
import data_processing.AudioProcessing as AP
from pathlib import Path
import kagglehub
import os
if __name__ == "__main__":
    downloader = SongDownloader()
    noise_path = Path(kagglehub.dataset_download("moazabdeljalil/back-ground-noise")).absolute()
    noise_list = []
    for root, dirs, files in os.walk(noise_path):
        for file in files:
            full_path = os.path.join(root, file)
            noise_list.append(os.path.abspath(full_path))
    training, num_training = downloader.train_df, len(downloader.train_df)
    test, num_test = downloader.test_df, len(downloader.test_df)
    val, num_val = downloader.val_df, len(downloader.val_df)
    nchannels = 2; sample_rate = 44100
    
    for t_clip in range(num_training):
        clip_path, f_type = downloader.get_clip(t_clip, "train")
        audio = AP.Utils.get_audio_and_rechannel(clip_path, nchannels)
        res_audio = AP.Utils.resample(audio=audio, new_sr= sample_rate)
        shift_audio = AP.Utils.time_shift(audio=res_audio, max_shift=0.4)
        noisy_audio = AP.Utils.add_noise(audio=shift_audio, noise_list=noise_list)
        sgram = AP.Utils.spectrogram(noisy_audio)
        aug_sgram = AP.Utils.augment_spectrogram(spec=sgram, sr=sample_rate, n_fmask=2, n_tmask=2)


    for d_clip in range(num_test):
        clip_path, f_type = downloader.get_clip(t_clip, "test")

    for v_clip in range(num_val):
        clip_path, f_type = downloader.get_clip(v_clip, "val")