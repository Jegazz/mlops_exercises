from src.data.data import mnist
from omegaconf import OmegaConf
import torch

def test_training():
    cfg = OmegaConf.load('src/models/configs/training_conf.yaml')
    batch_size = cfg.training.batch_size

    train_dataset, test_dataset = mnist()
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for images, labels in trainloader:
        assert images.shape[0] == labels.shape[0], f"Images and labels need to have the same batch size"

test_training()
        