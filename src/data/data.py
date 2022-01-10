import os

import numpy as np
import torch


def mnist():
    # path = os.path.abspath(os.path.join(os.getcwd(), 'data', 'raw', 'corruptmnist')) # used for debugging
    path = os.path.abspath(os.path.join(os.getcwd(), '..','..','..', 'data', 'raw', 'corruptmnist'))

    images = []
    labels = []
    for i in range(5):
        file_name = 'train_' + str(i) + '.npz'
        train_set = np.load(os.path.join(path, file_name))
        images.append(torch.Tensor(train_set[train_set.files[0]]))
        labels.append(torch.Tensor(train_set[train_set.files[1]]).type(torch.LongTensor))

    train_images = torch.cat(([t for t in images]), dim = 0)
    train_labels = torch.cat(([t for t in labels]), dim = 0)
    train_dataset = torch.utils.data.TensorDataset(train_images, train_labels)
    
    test_set = np.load(os.path.join(path, 'test.npz'))
    img = torch.Tensor(test_set[test_set.files[0]])
    lbs = torch.Tensor(test_set[test_set.files[1]]).type(torch.LongTensor)
    test_dataset = torch.utils.data.TensorDataset(img, lbs)
    return train_dataset, test_dataset