import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
import argparse
import logging
import sys
from pathlib import Path

import click
import hydra
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
from dotenv import find_dotenv, load_dotenv
from model import MyAwesomeConvolutionalModel
from model_pytorch_lightning import MyLitAwesomeConvolutionalModel
from omegaconf import OmegaConf
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

import wandb
from src.data.make_dataset import MNISTDataset

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

        train_images, train_labels = torch.load(orig_cwd+"/data/processed/train_images.pt"), torch.load(orig_cwd+"/data/processed/train_labels.pt")
        trainset = MNISTDataset(train_images, train_labels)
        trainloader = DataLoader(trainset, batch_size = params.batch_size)

        test_images, test_labels = torch.load(orig_cwd+"/data/processed/test_images.pt"), torch.load(orig_cwd+"/data/processed/test_labels.pt")
        testset = MNISTDataset(test_images, test_labels)
        testloader = DataLoader(testset, batch_size = params.batch_size)

        model = MyLitAwesomeConvolutionalModel(10, params.lr, params.dropout_p)

        trainer = pl.Trainer(max_epochs = params.epochs, 
                            limit_train_batches = 1.0, 
                            logger=pl.loggers.WandbLogger(project="MNIST_classifier"))
        trainer.fit(model, trainloader)
        trainer.test(dataloaders=testloader)

        torch.save(model.state_dict(), orig_cwd+'/models/convolutional/checkpoint.pth')

        # save_results(train_loss, orig_cwd)
        # fig = plt.figure(figsize=(8, 5))
        # plt.plot(train_loss)
        # plt.ylabel("Train loss")
        # plt.xlabel("Epochs")
        # plt.title("Learning curve - training")
        # plt.savefig(f"{orig_cwd}/reports/figures/Training_curve.png")
        # wandb.log({"plot": wandb.Image(fig)})

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