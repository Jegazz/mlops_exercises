import argparse
import os
import pdb
import sys
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from src.models.model import MyAwesomeModel
from numpy import mod
from torch import nn, optim
from src.data.data import mnist
import hydra

@hydra.main(config_path="configs", config_name="config.yaml")
def TrainModel(cfg):
    experiment_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = experiment_time + "_MyAwesomeModel" + ".pt"
    figure_name = experiment_time + "_MyAwesomeModel" + ".png"
    figure_path = os.path.abspath(os.path.join(os.getcwd(), '..', '..','..', 'reports', 'figures', figure_name))
    trained_models_path = os.path.abspath(os.path.join(os.getcwd(),'..', '..','..', 'models', model_name))

    # Loading parameters from config file
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs

    model = MyAwesomeModel(cfg.model)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Extracting datasets
    train_dataset, _ = mnist()
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    training_loss = []

    for e in range(epochs):
        epoch_loss = 0
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            outputs = model(images)
            # pdb.set_trace()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        training_loss.append(epoch_loss / len(trainloader))

        if e % 5 == 0:
            print(f"Epoch {e}, training loss: {training_loss[-1]}")

    # Plotting trainig curve
    epoch = np.arange(len(training_loss))
    plt.figure()
    plt.plot(
        epoch,
        training_loss,
        "b",
        label="Training loss",
    )
    plt.legend()
    plt.xlabel("Epoch"), plt.ylabel("MSE Loss")

    # Saving Training Curve under report/figures
    plt.savefig(figure_path)

    # Saving Model
    torch.save(model.state_dict(), trained_models_path)

if __name__ == "__main__":
    TrainModel()