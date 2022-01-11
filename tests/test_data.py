from src.data.data import mnist
import unittest

def test_data():
    train_dataset, test_dataset = mnist()

    # Asserting dataset lengths
    assert len(train_dataset) == 25000
    assert len(test_dataset) == 5000

    label_mess = "Label value outside range(0,10)"

    # Asserting shapes
    for images, label in train_dataset:
        assert images.shape == (28,28), "Image shape has to be [28, 28]"
        unittest.TestCase.assertTrue( 0 <= label.item() <= 10, label_mess, "Label value has to be in the range (0,10)")

test_data()
    