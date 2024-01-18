import torch
import torch.nn as nn

import hydra
import wandb
import logging
import sys
import os
import functools
import torcheval.metrics as torch_metrics
from dotenv import load_dotenv

from torch.optim import Adam
from data.make_dataset import fetch_dataloader

from models.modelTIMM import get_timm

from utils.logging import init_logging, with_default_logging
from utils.operations import to3channels
from utils.evaluation import evaluate


@with_default_logging(None)
def instantiate_training_objects(cfg, DEVICE):
    """
    Instantiating essential training objects for training
    
    ARGS:
        cfg: Hydra config object
        DEVICE: PyTorch device object
    
    RETURNS:
        model: PyTorch model
        train_dataloader: PyTorch dataloader
        validation_dataloader: PyTorch dataloader
        criterion: PyTorch loss function
        optimizer: PyTorch optimizer
    """
    # import timm_model here
    model = get_timm().to(DEVICE)
    
    train_dataloader, validation_dataloader = fetch_dataloader(DEVICE, **cfg.data_fetch)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.hyperparams.lr)
    log_msg = f"Fetched data: (Train: {len(train_dataloader)}, Val: {len(validation_dataloader)})"
    return (model, train_dataloader, validation_dataloader, criterion, optimizer), (log_msg, None)

#region Training loop
def train_batch(model, data_batch, optimizer, criterion, running_train_loss):
    """
    Train a model for one batch
    
    ARGS:
        model: PyTorch model
        data_batch: PyTorch dataloader
        optimizer: PyTorch optimizer
        criterion: PyTorch loss function
        running_train_loss: float
    RETURNS:
        model: PyTorch model
        optimizer: PyTorch optimizer
        criterion: PyTorch loss function
        running_train_loss: float
        train_loss: float
    """
    images, labels = data_batch
    images = to3channels(images)
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    preds = model(images)

    loss = criterion(preds, labels)
    loss.backward()
    train_loss = loss.item()
    running_train_loss += train_loss

    optimizer.step()

    return model, optimizer, criterion, running_train_loss, train_loss


@with_default_logging(None)
def train_step(model, train_dataloader, optimizer, criterion, cfg, **kwargs):
    """
    Trains a model for one epoch
    
    ARGS:
        model: PyTorch model
        train_dataloader: PyTorch dataloader
        optimizer: PyTorch optimizer
        criterion: PyTorch loss function
        
    **KWARGS: 
        cfg: Hydra config object
        epoch: int
    
    RETURNS: 
        model: PyTorch model
        optimizer: PyTorch optimizer
        running_train_loss: float
        last_train_loss: float
        log_msg: str
        wandb_data: dict
    """
    
    EPOCH : int = kwargs.get("epoch", 0)
    running_train_loss = kwargs.get("running_train_loss", 0)
    

    #Prepares model for training
    model.train()

    for batch_idx, data_batch in enumerate(train_dataloader):
        model, optimizer, criterion, running_train_loss, last_train_loss = train_batch(model, data_batch, optimizer, criterion, running_train_loss)
        
            
    # Done training
    log_msg = f'--- Epoch {EPOCH+1}/{cfg.hyperparams.epochs} | train loss: {last_train_loss} ---'
    wandb_data = None

    return (model, optimizer, running_train_loss), (log_msg, wandb_data)


@with_default_logging(None)
def validate_step(model, validation_dataloader, criterion, validation_loss = 0):
    """
    Validate a model given the model and a validation dataloader
    
    ARGS:
        model: PyTorch model
        validation_dataloader: PyTorch dataloader
        criterion: PyTorch loss function
        validation_loss: float
    
    RETURNS:
        accuracy: float
        validation_loss: float
    """
    
    model.eval()
    with torch.no_grad():
        correct_predictions = 0
        total_samples = 0
        for batch_idx, (images, labels) in enumerate(validation_dataloader):                
            images = to3channels(images)

            preds = model(images)
            
            loss = criterion(preds, labels)
            validation_loss += loss
            
            correct_predictions += torch.sum(preds.argmax(dim=1) == labels).item()
            total_samples += len(labels)

    accuracy = correct_predictions / total_samples
    
    log_msg = f'--- | accuracy: {accuracy} | val loss: {validation_loss / len(validation_dataloader)} ---\n'
    wandb_data = {
        "accuracy" : accuracy,
        "validation_loss" : validation_loss / len(validation_dataloader)
    }
    
    return (accuracy, validation_loss), (log_msg, wandb_data)

@with_default_logging("Start training")
def train_loop(model, train_dataloader, validation_dataloader, criterion, optimizer, cfg):
    for epoch in range(cfg.hyperparams.epochs):
        running_train_loss = 0
        last_train_loss = 0
        validation_loss = 0

        # Start training step
        kwargs = {
            "cfg": cfg,
            "epoch": epoch,
            "running_train_loss": running_train_loss,
        }
        model, optimizer, running_train_loss = train_step(model, train_dataloader, optimizer, criterion, **kwargs)
        
        # Start validation step
        accuracy, validation_loss = validate_step(model, validation_dataloader, criterion, validation_loss)

    log_msg = f"Finished training. Accuracy: {accuracy}"
    return model, (log_msg, None)
#endregion


            

    

def save_model(model, cfg):
    if os.path.isdir(f"/gcs/{cfg.data_fetch.gcp_bucket_name}"):
        torch.save(model.state_dict(), f"/gcs/{cfg.data_fetch.gcp_bucket_name}/models/model_{cfg.base.experiment_name}.pt") # saves to gcp
    # Saves when working locally, not in a container
    elif os.path.isdir("models/"): 
        torch.save(model.state_dict(), f"models/model_{cfg.base.experiment_name}.pt")


    
    


@hydra.main(config_path="config", config_name="train_model.yaml",version_base='1.3')
def main(cfg):
    """ 
    Train the model 
    """
    
    # Instantiate wandb and the logging object
    init_logging(cfg)
    
    # Instatiate device
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    
    # Instantiate training objects
    #training objects = (model, train_dataloader, validation_dataloader, criterion, optimizer)
    training_objects = instantiate_training_objects(cfg, DEVICE)
    
    model = train_loop(*training_objects, cfg)
    
    evaluate(model, training_objects[2], training_objects[3], cfg)
    
    save_model(model, cfg)
    
    wandb.finish()
    

if __name__ == '__main__':
    
    main()