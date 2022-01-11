from src.data.data import mnist
import unittest
import os
import torch

def test_data():
    processed_train_name = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/train_dataset.pt'))
    processed_test_name = os.path.abspath(os.path.join(os.getcwd(), 'data/processed/test_dataset.pt'))

    train_dataset = torch.load(processed_train_name)
    test_dataset = torch.load(processed_test_name)

    # Asserting dataset lengths
    assert len(train_dataset) == 25000
    assert len(test_dataset) == 5000

    label_mess = "Label value outside range(0,10)"

    # Asserting shapes
    for images, label in train_dataset:
        assert images.shape == (28,28), "Image shape has to be [28, 28]"
        unittest.TestCase.assertTrue( 0 <= label.item() <= 10, label_mess, "Label value has to be in the range (0,10)")

if __name__ == "__main__":
    test_data()
    