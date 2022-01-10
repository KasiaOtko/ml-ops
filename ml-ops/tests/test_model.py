import pytest
import torch
from torch.utils.data import DataLoader

from src.data.make_dataset import MNISTDataset
from src.models.model import MyAwesomeConvolutionalModel


def test_model_input_output():

    batch_size = 64
    train_images, train_labels = torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_labels.pt")
    trainset = MNISTDataset(train_images, train_labels)
    trainloader = DataLoader(trainset, batch_size = batch_size)

    last_batch_size = 25000 % batch_size
    model = MyAwesomeConvolutionalModel(10)
    for step, (images, labels) in enumerate(trainloader):
        # if this is not the last batch
        if step != len(trainloader)-1:
            # check for the dimension of input
            assert images.shape == torch.Size([batch_size, 1, 28, 28])
            assert labels.shape == torch.Size([batch_size])
            log_ps = model(images.float())
            # check for the dimension of output
            assert log_ps.shape == torch.Size([batch_size, 10])
        else:
            # check for the dimension of input
            assert images.shape == torch.Size([last_batch_size, 1, 28, 28])
            assert labels.shape == torch.Size([last_batch_size])
            log_ps = model(images.float())
            # check for the dimension of output
            assert log_ps.shape == torch.Size([last_batch_size, 10])

def test_on_wrong_shape_to_forward():
    model = MyAwesomeConvolutionalModel(10)
    with pytest.raises(ValueError, match = "Expected input is not a 4D tensor."):
        model(torch.randn(1, 2, 3))

if __name__ == '__main__':
    test_on_wrong_shape_to_forward()
    test_model_input_output()

    




