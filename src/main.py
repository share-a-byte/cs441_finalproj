from src import Model, Loader
from torch.utils.data import random_split

if __name__ == "__main__":
	train_dataloader = Loader.SoundDataset(mode=0); val_dataloader = Loader.SoundDataset(mode=1); test_dataloader = Loader.SoundDataset(mode=2)
	model = Model.AudioCNN()
	Model.train_model(model, train_loader=train_dataloader, val_loader=val_dataloader, num_epochs=100, lrt=0.01)
	
