import torch
import torch.nn as nn
import hydra
from torch.optim import Adam
from torch.utils.data import DataLoader
from data.make_dataset import fetch_dataloader

#import wandb
import logging

from models.model import Net

@hydra.main(config_path="config", config_name="train_model.yaml",version_base='1.3')
def train(cfg):
    """ 
    Train the model 
    
    Return: none
    """
    print("Training the model")

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR = cfg.hyperparams.lr
    model = Net().to(DEVICE)
    model.train()
    train_set, _ = fetch_dataloader(cfg.data_fetch)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    epoch = 5
    print("Start training")
    for epoch in range(epoch):
        running_loss = 0
        for batch_idx, (images, labels) in enumerate(train_set):
            # Flatten MNIST images into a 784 long vector
            images = images.view(images.shape[0], -1)
        
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()


        else:
            print(f"Training loss: {running_loss}")

if __name__ == '__main__':
    train()


    
