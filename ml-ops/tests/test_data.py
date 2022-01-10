import torch

from src.data.make_dataset import MNISTDataset

def test_data():
    train_images, train_labels = torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_labels.pt")
    trainset = MNISTDataset(train_images, train_labels)

    test_images, test_labels = torch.load("data/processed/test_images.pt"), torch.load("data/processed/test_labels.pt")
    testset = MNISTDataset(test_images, test_labels)
    N_train = 25000
    N_test = 5000

    assert len(trainset) == N_train
    assert len(testset) == N_test
    # Check dimension of images
    assert trainset.images.shape[1:] == torch.Size([1, 28, 28])
    assert testset.images.shape[1:] == torch.Size([1, 28, 28])
    # Check that all labels are represented
    assert all(i in trainset.labels for i in range(10))
    assert all(i in testset.labels for i in range(10))

if __name__ == '__main__':
    test_data()