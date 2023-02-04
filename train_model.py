# Required Imports

import os
import logging
import sys

#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms

import argparse

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# define logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addFilter(logging.StreamHandler(sys.stdout))


def test(model, test_loader, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1, keepdim=True)
            running_corrects += pred.eq(labels.view_as(pred)).sum().item()

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects / len(test_loader.dataset)
    print(f"Testing Accuracy: {100 * test_acc:.2f}%, Test Loss: {test_loss:.4f}")


def train(model, train_loader, criterion, optimizer):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        _, pred = outputs.argmax(dim=1, keepdim=True)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += pred.eq(labels.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = running_corrects / len(train_loader.dataset)
    
    
def net():
    
    # Function initializes model using a pretrained model

    model = models.resnet50(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
    
    num_features = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 256),
    nn.ReLU(inplace=True),
    nn.Linear(256, 133),
    nn.ReLU(inplace=True))
    return model

def create_data_loaders(data, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    train_dataset_path = os.path.join(data, "train")
    test_dataset_path = os.path.join(data, "test")

    training_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, 224),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, 224),
        transforms.ToTensor() 
    ])

    train_dataset = torchvision.datasets.ImageFolder(train_dataset_path, transform=training_transform)
    test_dataset = torchvision.datasets.ImageFolder(test_dataset_path, transform=testing_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    return train_loader, test_loader

def main(args):
    '''
    TODO: Initialize a model by calling the net function
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=net()
    model.to(device)
    logger.info(f"Device: {device}")
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=args.lr, momentum=args.momentum)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model=train(model, train_loader, loss_criterion, optimizer)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='learning rate (default: 0.01)',
        metavar='LR'
    )

    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.9)',
        metavar='M')

    parser.add_argument(
        '--epochs',
        type=int,
        default=1,
        help='number of epochs to train (default: 10)',
        metavar='N'
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='input batch size for training (default: 64)',
        metavar='N'
    )

    args=parser.parse_args()
    
    main(args)