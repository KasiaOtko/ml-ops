import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import logging
import sys
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeConvolutionalModel
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.data.make_dataset import MNISTDataset


@click.command()
@click.argument('lr', default = 0.01, type=float)
@click.argument('epochs', default = 10, type=int)
def train(lr, epochs):
        print("Training day and night")
        print(f"Learning rate: {lr}, Epochs: {epochs}")
        # parser = argparse.ArgumentParser(description='Training arguments')
        # parser.add_argument('--lr', default=0.1)
        # parser.add_argument('--epochs', default=10)
        # add any additional argument that you want
        # args = parser.parse_args(sys.argv[3:])
        # args = vars(args)
        # print(args)

        # TODO: Implement training loop here
        model = MyAwesomeConvolutionalModel(10)
        #trainset, _ = torch.load("data/processed/trainset.pt")

        train_images, train_labels = torch.load("data/processed/train_images.pt"), torch.load("data/processed/train_labels.pt")
        trainset = MNISTDataset(train_images, train_labels)
        trainloader = DataLoader(trainset, batch_size = 64)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        epochs = epochs
        train_loss = []
        for e in range(epochs):
            batch_loss = []
            for images, labels in trainloader:

                log_ps = model(images.float())
                #print(labels.long())
                loss = criterion(log_ps, labels.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
    
            train_loss.append(np.mean(batch_loss))
            print(f"Epoch {e}, Train loss: {train_loss[e]}")

        print(model)
        torch.save(model.state_dict(), 'models/convolutional/checkpoint.pth')

        save_results(train_loss)

        return model

def save_results(train_loss: list):
    '''
    Saves learning curve for a training loop.
        Parameters: 
            train_loss (list): list of training losses per epoch

        Returns:
            nothing, instead saves the figure to "reports/figures/Training_curve.png"
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss)
    plt.ylabel("Train loss")
    plt.xlabel("Epochs")
    plt.title("Learning curve - training")
    plt.savefig("reports/figures/Training_curve.png")

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()