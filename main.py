import argparse
import sys
import numpy as np

import torch
from torch import nn, optim

from data import mnist
from model import MyAwesomeModel, MyAwesomeConvolutionalModel


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.1)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        args = vars(args)
        print(args['lr'])
        # TODO: Implement training loop here
        #model = MyAwesomeModel(784, 256, 128, 10)
        model = MyAwesomeConvolutionalModel(10)
        train_loader, _ = mnist()

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=float(args['lr']))

        epochs = 10
        train_loss = []
        for e in range(epochs):
            batch_loss = []
            for images, labels in train_loader:
            
                log_ps = model(images.float())
                
                loss = criterion(log_ps, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())
    
            train_loss.append(np.mean(batch_loss))
            print(f"Epoch {e}, Train loss: {train_loss[e]}")

        print(model)
        torch.save(model.state_dict(), 'checkpoint.pth')

        return model
        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        # model = MyAwesomeModel(784, 256, 128, 10)
        model = MyAwesomeConvolutionalModel(10)
        state_dict = torch.load(args.load_model_from)
        model.load_state_dict(state_dict)
        print(model)

        _, test_loader = mnist()

        criterion = nn.CrossEntropyLoss()

        accuracy = 0
        test_loss = 0

        for images, labels in test_loader:
    
            #images = images.resize_(images.size()[0], 784)

            output = model(images.float())
            test_loss += criterion(output, labels).item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(output)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            accuracy += equality.type_as(torch.FloatTensor()).mean()

        print(accuracy)

if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    