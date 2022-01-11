import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeConvolutionalModel
from torch import nn
from torch.utils.data import DataLoader

from src.data.make_dataset import MNISTDataset


@click.command()
@click.argument('load_model_from', type=click.Path(exists=True))
def predict(load_model_from):
    model = MyAwesomeConvolutionalModel(10)
    state_dict = torch.load(load_model_from)
    model.load_state_dict(state_dict)

    test_images, test_labels = torch.load("data/processed/test_images.pt"), torch.load("data/processed/test_labels.pt")
    testset = MNISTDataset(test_images, test_labels)
    testloader = DataLoader(testset, batch_size=64)

    criterion = nn.CrossEntropyLoss()

    batch_loss = []
    accuracy = []
    for images, labels in testloader:

        # images = images.resize_(images.size()[0], 784)

        output = model(images.float())
        batch_loss.append(criterion(output, labels.long()).item())

        # Calculating the accuracy
        # Model's output is log-softmax, take exponential to get the probabilities
        ps = torch.exp(output)
        # Class with highest probability is our predicted class, compare with true label
        equality = (labels.data == ps.max(1)[1])
        accuracy.append(equality.type_as(torch.FloatTensor()).mean())

    print(f"Accuracy: {np.mean(accuracy)}")
    print("Test loss", np.mean(batch_loss))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    predict()
