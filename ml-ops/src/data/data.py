import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PATH = "corruptmnist/"


def mnist():
    # read image files
    test = np.load(PATH + "\\test.npz")
    train0 = np.load(PATH + "\\train_0.npz")
    train1 = np.load(PATH + "\\train_1.npz")
    train2 = np.load(PATH + "\\train_2.npz")
    train3 = np.load(PATH + "\\train_3.npz")
    train4 = np.load(PATH + "\\train_4.npz")

    # extract images and labels
    train_images = np.concatenate(
        (
            train0["images"],
            train1["images"],
            train2["images"],
            train3["images"],
            train4["images"],
        )
    )
    train_labels = np.concatenate(
        (
            train0["labels"],
            train1["labels"],
            train2["labels"],
            train3["labels"],
            train4["labels"],
        )
    )
    test_images = test["images"]
    test_labels = test["labels"]

    # train_images = torch.Tensor(train_images)
    # train_labels = torch.Tensor(train_labels)
    # test_images = torch.Tensor(test_images)
    # test_labels = torch.Tensor(test_labels)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )

    trainset = MNISTDataset(train_images, train_labels, transform=transform)
    trainloader = DataLoader(trainset, batch_size=64)
    testset = MNISTDataset(test_images, test_labels, transform=transform)
    testloader = DataLoader(testset, batch_size=64)
    return trainloader, testloader


class MNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.transform:
            images_out = self.transform(self.images[idx])

        return images_out, self.labels[idx]
