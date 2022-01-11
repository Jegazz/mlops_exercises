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

def EvaluateModel():
    parser = argparse.ArgumentParser(
        description="Script for training models",
        usage="python predict_model.py <arguments>",
    )
    parser.add_argument("load_model_from", default="")
    args = parser.parse_args(sys.argv[1:])

    model = MyAwesomeModel()
    state_dict = torch.load(
        os.path.join(os.getcwd(), 'models', args.load_model_from)
    )

    model.load_state_dict(state_dict)
    model.eval()

    criterion = nn.NLLLoss()
    batch_size = 64

    _, test_dataset = mnist()
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )

    with torch.no_grad():
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            # Accuracy
            ps = torch.exp(outputs)
            equality = labels.data == ps.max(1)[1]
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print("Accuracy: {:2f}%".format((accuracy / len(testloader)) * 100))


if __name__ == "__main__":
    #TrainOREvaluate()
    EvaluateModel()