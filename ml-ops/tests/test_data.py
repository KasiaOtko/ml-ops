import torch
import pytest

from src.data.make_dataset import MNISTDataset

import os.path

@pytest.mark.skipif(not os.path.exists("data/processed/train_images.pt") 
                        or not os.path.exists("data/processed/train_labels.pt"), reason="Train images or labels files not found")
@pytest.mark.skipif(not os.path.exists("data/processed/test_images.pt") 
                        or not os.path.exists("data/processed/test_labels.pt"), reason="Test images or labels files not found")
@pytest.mark.parametrize("test_input, expected", [("len(trainset)", 25000), ("len(testset)", 5000)])
def test_data(test_input, expected):

    train_images, train_labels = torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_labels.pt")
    trainset = MNISTDataset(train_images, train_labels)

    test_images, test_labels = torch.load("data/processed/test_images.pt"), torch.load("data/processed/test_labels.pt")
    testset = MNISTDataset(test_images, test_labels)

    assert eval(test_input) == expected, "Train or test dataset did not have the correct number of samples"
    # Check dimension of images
    assert trainset.images.shape[1:] == torch.Size([1, 28, 28]), "Shape of train images is incorrect"
    assert testset.images.shape[1:] == torch.Size([1, 28, 28]), "Shape of test images is incorrect"
    # Check that all labels are represented
    assert all(i in trainset.labels for i in range(10)), "Not all of the classes are represented in the train set"
    assert all(i in testset.labels for i in range(10)), "Not all of the classes are represented in the test set"

if __name__ == '__main__':
    test_data()