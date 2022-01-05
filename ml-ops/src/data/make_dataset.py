# -*- coding: utf-8 -*-
import logging
from pathlib import Path

import click
import numpy as np
import torch
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms

#from src.models.train_model import train


@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')

    train_images, train_labels, test_images, test_labels = mnist(input_filepath)

    torch.save(train_images, output_filepath + '/train_images.pt')
    torch.save(train_labels, output_filepath + '/train_labels.pt')
    torch.save(test_images, output_filepath + '/test_images.pt')
    torch.save(test_labels, output_filepath + '/test_labels.pt')

    print("Successfully saved.")


def mnist(input_filepath):
    # read image files
    test = np.load(input_filepath + "/test.npz")
    train0 = np.load(input_filepath + "/train_0.npz")
    train1 = np.load(input_filepath + "/train_1.npz")
    train2 = np.load(input_filepath + "/train_2.npz")
    train3 = np.load(input_filepath + "/train_3.npz")
    train4 = np.load(input_filepath + "/train_4.npz")
    
    # extract images and labels
    train_images = np.concatenate((train0['images'], train1['images'], train2['images'], train3['images'], train4['images']))
    train_labels = np.concatenate((train0['labels'], train1['labels'], train2['labels'], train3['labels'], train4['labels']))
    test_images = test['images']
    test_labels = test['labels']

    # convert to tensors
    train_images = torch.Tensor(train_images)
    train_labels = torch.Tensor(train_labels)
    test_images = torch.Tensor(test_images)
    test_labels = torch.Tensor(test_labels)

    # normalize
    preprocess_transform = transforms.Normalize((0.5,), (0.5,))
    train_images = preprocess_transform(train_images)
    test_images = preprocess_transform(test_images)
    train_images = train_images[:, None, :, :]
    test_images = test_images[:, None, :, :]
    #print(test_images.shape)

    # store images and labels in one dictionary
    # trainset = dict()
    # trainset['images'] = train_images
    # trainset['labels'] = train_labels

    # testset = dict()
    # testset['images'] = test_images
    # testset['labels'] = test_labels
    # trainset = MNISTDataset(train_images, train_labels, transform = transform)
    # trainloader = DataLoader(trainset, batch_size = 64)
    # testset = MNISTDataset(test_images, test_labels, transform = transform)
    # testloader = DataLoader(testset, batch_size = 64)
    return train_images, train_labels, test_images, test_labels

class MNISTDataset(TensorDataset):
    def __init__(self, images, labels, transform = None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_in = self.images[idx]
        img_out = img_in.clone()
        labels = self.labels[idx]

        if self.transform:
            img_out = self.transform(img_out)

        return img_out, labels

if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
