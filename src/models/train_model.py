import os

import logging
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeConvolutionalModel
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader

import wandb
from src.data.make_dataset import MNISTDataset

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

wandb.init(project="MNIST_classifier", entity="thekatin")

hydra.output_subdir = None


# @click.command()
# @click.argument('lr', default = 0.01, type=float)
# @click.argument('epochs', default = 10, type=int)
@hydra.main(config_path="config", config_name="training_config.yaml")
def train(config):

    wandb.config = config.hyperparams
    orig_cwd = hydra.utils.get_original_cwd()
    orig_cwd = orig_cwd.replace(os.sep, '/')
    print(f"configuration: \n {OmegaConf.to_yaml(config)}")
    params = config.hyperparams
    # parser = argparse.ArgumentParser(description='Training arguments')
    # parser.add_argument('--lr', default=0.1)
    # parser.add_argument('--epochs', default=10)
    # args = parser.parse_args(sys.argv[3:])
    # args = vars(args)
    # print(args)

    model = MyAwesomeConvolutionalModel(10)

    train_images, train_labels = torch.load(orig_cwd + "/data/processed/train_images.pt"), torch.load(orig_cwd + "/data/processed/train_labels.pt")
    trainset = MNISTDataset(train_images, train_labels)
    trainloader = DataLoader(trainset, batch_size=params.batch_size)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params.lr)

    print("Training day and night")
    train_loss = []
    accuracy = []
    for e in range(params.epochs):
        batch_loss = []
        accuracy_batch = []
        for images, labels in trainloader:

            log_ps = model(images.float())

            loss = criterion(log_ps, labels.long())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss.append(loss.item())
            ps = torch.exp(log_ps)
            equality = (labels.data == ps.max(1)[1]).type_as(torch.FloatTensor()).mean()
            accuracy_batch.append(equality)

        train_loss.append(np.mean(batch_loss))
        accuracy.append(np.mean(accuracy_batch))
        wandb.log({"loss": train_loss[e]})
        print(f"Epoch {e}, Train loss: {train_loss[e]}, Accuracy: {accuracy[e]}")

    # print(model)
    torch.save(model.state_dict(), orig_cwd + '/models/convolutional/checkpoint.pth')

    # save_results(train_loss, orig_cwd)
    fig = plt.figure(figsize=(8, 5))
    plt.plot(train_loss)
    plt.ylabel("Train loss")
    plt.xlabel("Epochs")
    plt.title("Learning curve - training")
    plt.savefig(f"{orig_cwd}/reports/figures/Training_curve.png")
    wandb.log({"plot": wandb.Image(fig)})

    return model


def save_results(train_loss: list, orig_cwd):
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
    plt.savefig(f"{orig_cwd}/reports/figures/Training_curve.png")


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    train()
