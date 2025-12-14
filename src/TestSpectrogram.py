import data_processing.AudioProcessing as AP
import os
#should fix conda environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from pathlib import Path
import random
import kagglehub
import matplotlib.pyplot as plt
import numpy
import torch

def get_random_file(directory):
    files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
    if not files:
        return None
    return random.choice(files)

def display_spectrogram(spec):
    print('Spectrogram tensor has shape {}'.format(spec.shape))
    plt.figure(figsize=(10, 4))
    plt.imshow(spec[0].numpy(), origin="lower", aspect="auto", cmap="viridis")
    plt.colorbar(label="dB")
    plt.title("Spectrogram")
    plt.xlabel("Time")
    plt.ylabel("Frequency Bin")
    plt.show()
    
def display_waveform(audio):
    wave, sr = audio
    wave = wave.numpy()
    nchannels, nframes = wave.shape
    time_ax = torch.arange(0, nframes) / sr
    figure, axes = plt.subplots(nchannels, 1, sharex=True)
    if nchannels == 1:
        axes = [axes]
    for c in range(nchannels):
        axes[c].plot(time_ax, wave[c], linewidth=1)
        axes[c].grid(True)
        axes[c].set_ylabel(f"Channel {c+1}")
    
    plt.xlabel("Time (s)")
    plt.show()

if __name__ == "__main__":
    noise_path = Path(kagglehub.dataset_download("moazabdeljalil/back-ground-noise")).absolute()
    noise_list = []
    for root, dirs, files in os.walk(noise_path):
        for file in files:
             full_path = os.path.join(root, file)
             print(full_path)
             noise_list.append(os.path.abspath(full_path))
    sr = 44100
    shift_pct = 0.2
    max_dur = 5
    clip_path = "./video_clipping/clips/5sec/not_AI/Bob Marley & The Wailers - Three Little Birds (Official Music Video) [HNBCVM4KbUM]_005.mp3"
    aud = AP.Utils.get_audio_and_rechannel(clip_path, 2)
    reaud = AP.Utils.resample(aud, sr)
    display_waveform(reaud)
    shift_aud = AP.Utils.time_shift(reaud, shift_pct)
    display_waveform(shift_aud)
    noise_aud = AP.Utils.add_noise(shift_aud, noise_list=noise_list)
    display_waveform(noise_aud)
    sgram = AP.Utils.spectrogram(noise_aud)
    display_spectrogram(sgram)
    aug_sgram = AP.Utils.augment_spectrogram(sgram, sr=sr, max_mask=0.1, n_fmask=2, n_tmask=2)
    display_spectrogram(aug_sgram)
    
    
