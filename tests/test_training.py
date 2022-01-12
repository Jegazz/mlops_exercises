from src.data.data import mnist
from omegaconf import OmegaConf
import torch
import os

def test_training():
    cfg = OmegaConf.load('src/models/configs/training_conf.yaml')
    batch_size = cfg.training.batch_size
    
    # Create artificial train dataset
    train_images = torch.rand(25000,28,28)
    train_labels = torch.randint(0,1,(25000,1))
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)

    for images, labels in trainloader:
        assert images.shape[0] == labels.shape[0], f"Images and labels need to have the same batch size"

if __name__ == "__main__":
    test_training()
        