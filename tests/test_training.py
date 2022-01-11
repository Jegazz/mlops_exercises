from src.data.data import mnist
from omegaconf import OmegaConf
import torch
import os

def test_training():
    cfg = OmegaConf.load('src/models/configs/training_conf.yaml')
    batch_size = cfg.training.batch_size

    processed_train_name = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/train_dataset.pt'))
    train_dataset = torch.load(processed_train_name)

    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for images, labels in trainloader:
        assert images.shape[0] == labels.shape[0], f"Images and labels need to have the same batch size"

test_training()
        