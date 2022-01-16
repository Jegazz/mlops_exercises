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
from omegaconf import OmegaConf
from torch.profiler import profile, record_function, ProfilerActivity
import subprocess


def TrainModel(cfg):
    experiment_time = time.strftime("%Y%m%d-%H%M%S")
    model_name = experiment_time + "_MyAwesomeModel" + ".pt"
    figure_name = experiment_time + "_MyAwesomeModel" + ".png"
    figure_path = os.path.abspath(os.path.join(os.getcwd(), 'reports', 'figures', figure_name))
    trained_models_path = os.path.abspath(os.path.join(os.getcwd(),'models', model_name))
    processed_train_name = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/train_dataset.pt'))

    # Loading training parameters from config file
    learning_rate = cfg.training.learning_rate
    batch_size = cfg.training.batch_size
    epochs = cfg.training.epochs
    
    model = MyAwesomeModel()
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Extracting datasets
    train_dataset = torch.load(processed_train_name)
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    training_loss = []

    for e in range(epochs):
        epoch_loss = 0
        model.train()
        for images, labels in trainloader:
            optimizer.zero_grad()
            #with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            #    with record_function("model_inference"):
            #        outputs = model(images)
            #print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=30))
            outputs = model(images)
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

    #Saving model in the bucket
    tmp_model_file = os.path.join('/tmp', model_name)
    torch.save(model.state_dict(), tmp_model_file)
    subprocess.check_call(['gsutil', 'cp', tmp_model_file, os.path.join('/trained_models', model_name)])

if __name__ == "__main__":
    cfg = OmegaConf.load('src/models/configs/training_conf.yaml')
    TrainModel(cfg)