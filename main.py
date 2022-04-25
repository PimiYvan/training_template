from operator import mod
from numpy import require
import torch
from torch.utils.data import DataLoader,random_split
from data.dataloader import CustomDataSet
from config import args
from models import resnet, CNN
from utils import transform, img_display
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch.nn as nn
from train import train
from utils import plot

if  __name__ == '__main__':
    print('hello world of our template')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    metadata = args.metadata
    path = args.path
    train_size = args.train_size
    batch_size = args.bs

    data = CustomDataSet(metadata, path, transform=transform)

    train_size= int(train_size*len(data))
    val_size = len(data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(data, [train_size, val_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    #Model 
    # import argparse
    parser = argparse.ArgumentParser(prog='PROG')
    parser.add_argument('-m', '--model_name', help='This is the name of the model', required=True)
    parser.add_argument('-n', '--num_epochs',  help='This is the number of epoch', type=int, required=True)
    # args = parser.parse_args(['--model_name', '--num_epochs'])

    mains_args = vars(parser.parse_args())
    
    model_name = mains_args['model_name']
    num_epochs = mains_args['num_epochs']

    print(num_epochs, model_name)

    if model_name.lower() == 'resnet':
        model = resnet()
    elif model_name.lower() == 'cnn':
        model = CNN()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    criterion= nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=1.0)
    model_trained, percent, val_loss, val_acc, train_loss, train_acc= train(model, criterion, train_loader, val_loader, optimizer, num_epochs, device)

    plot(train_loss=train_loss, val_loss=val_loss)