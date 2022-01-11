import torch
from torch import nn
from src.models.model import MyAwesomeModel

def test_model():
    test_tensor = torch.rand(28,28).unsqueeze(0)
    assert test_tensor.shape == (1,28,28)

    model = MyAwesomeModel()
    model.eval()

    with torch.no_grad():
        output = model(test_tensor)
        assert output.shape == (1,10), "Output shape has to be [1, 10]"

if __name__ == "__main__":
    test_model()
