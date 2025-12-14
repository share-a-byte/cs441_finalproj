from model import AudioCNN, train_model
from loader import SoundDataset, spec_gram_collate
from pathlib import Path
from torch.utils.data import DataLoader
import kagglehub
import os

if __name__ == "__main__":
    noise_path = Path(kagglehub.dataset_download("moazabdeljalil/back-ground-noise")).absolute()
    noise_list = []
    for root, dirs, files in os.walk(noise_path):
        for file in files:
            full_path = os.path.join(root, file)
            noise_list.append(os.path.abspath(full_path))

    train_dataloader = DataLoader(dataset=SoundDataset(noise_list=noise_list, set_type="train"), batch_size=32, shuffle=True, collate_fn=spec_gram_collate)
    val_dataloader = DataLoader(dataset=SoundDataset(noise_list=noise_list, set_type="val"), batch_size=32, shuffle=False, collate_fn=spec_gram_collate)
    test_dataloader = DataLoader(dataset=SoundDataset(noise_list=noise_list, set_type="test"), batch_size=32, shuffle=False, collate_fn=spec_gram_collate)

    model = AudioCNN()
    trained_model = train_model(model, train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=10, lr=0.01)
	
