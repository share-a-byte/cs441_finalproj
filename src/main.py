from model import AudioCNN, train_model
from loader import SoundDataset
from pathlib import Path
import kagglehub
import os

if __name__ == "__main__":
    noise_path = Path(kagglehub.dataset_download("moazabdeljalil/back-ground-noise")).absolute()
    noise_list = []
    for root, dirs, files in os.walk(noise_path):
        for file in files:
            full_path = os.path.join(root, file)
            print(full_path)
            noise_list.append(os.path.abspath(full_path))

    train_dataloader = SoundDataset(noise_list=noise_list, set_type="train")
    val_dataloader = SoundDataset(noise_list=noise_list, set_type="val")
    test_dataloader = SoundDataset(noise_list=noise_list, set_type="test")

    model = AudioCNN()
    trained_model = train_model(model, train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=100, lr=0.01)
	
