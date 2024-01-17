import torch
import torch.nn as nn

import hydra
import wandb
import logging
import sys
import os
from dotenv import load_dotenv

from torch.optim import Adam
from data.make_dataset import fetch_dataloader

from models.model import Net
from models.modelTIMM import get_timm

def to3channels(images : torch.Tensor) -> torch.Tensor:
    """
    Duplicates the 1 channel image to a 3 channeled image by concatenating it to itself 3 times
    
    ARGS: Tensor of shape (batch_size, 1, 28, 28)
    
    RETURNS: Tensor of shape (batch_size, 3, 28, 28)
    """
    images = images.unsqueeze(1)
    images = torch.cat([images, images, images], dim=1) # convert to 3 channels (RGB)
    return images

def log_message(logger, msg : str, wandb_data : dict):
    """
    Logs a message to the logger and sends metadata to wandb
    
    ARGS:
        logger: logging object
        msg: str
        wandb_data: dict
        
    Returns: None
    """
    # Log to wandb
    logger.info(msg)
    wandb.log(wandb_data)

def train_step(model, train_dataloader, optimizer, criterion, **kwargs):
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
        logger: logging object
    
    RETURNS: None
    """
    
    cfg = kwargs.get("cfg", None)
    EPOCH : int = kwargs.get("epoch", 0)
    LOGGER = kwargs.get("logger", None)
    running_train_loss = kwargs.get("running_train_loss", 0)
    
    if cfg:
        LOGGING_TOGGLE : bool = cfg.base.logging_toggle
        LOGGING_INTERVAL : int = cfg.base.log_interval
        N_EPOCHS : int = cfg.hyperparams.epochs
    else:
        LOGGING_TOGGLE : bool = False
        LOGGING_INTERVAL : int = 5
        N_EPOCHS : int = 1    

    #Prepares model for training
    model.train()

    for batch_idx, data_batch in enumerate(train_dataloader):
        train_batch(model, data_batch, optimizer, criterion, running_train_loss)
        
        if not LOGGING_TOGGLE:
            continue
        if batch_idx % LOGGING_INTERVAL == LOGGING_INTERVAL - 1:
            last_train_loss = running_train_loss / LOGGING_INTERVAL
            running_train_loss = 0
            
            log_msg = f"epoch: {EPOCH+1}/{N_EPOCHS} | batch: {batch_idx+1}/{len(train_dataloader)} | loss: {last_train_loss}"
            wandb_data = {"train_loss": last_train_loss}
            log_message(LOGGER, log_msg, wandb_data)

    return model, optimizer, running_train_loss


def train_batch(model, data_batch, optimizer, criterion, running_train_loss):
    """
    Train a model for one batch
    
    ARGS:
        model: PyTorch model
        data_batch: PyTorch dataloader
        optimizer: PyTorch optimizer
        criterion: PyTorch loss function
        running_train_loss: float
    """
    images, labels = data_batch
    images = to3channels(images)
    optimizer.zero_grad()

    # Forward pass, then backward pass, then update weights
    preds = model(images)

    loss = criterion(preds, labels)
    loss.backward()
    running_train_loss += loss.item()

    optimizer.step()

    return model, optimizer, criterion, running_train_loss

    
        

        
        

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
    
    return accuracy, validation_loss

def evaluate():
    pass


@hydra.main(config_path="config", config_name="train_model.yaml",version_base='1.3')
def train(cfg) -> None:
    """ 
    Train the model 
    """

    # Create super basic logger
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    logger = logging.getLogger(__name__)

    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)

    # import timm_model here
    model = get_timm().to(DEVICE)


    train_dataloader, validation_dataloader = fetch_dataloader(DEVICE, **cfg.data_fetch)
    logger.info(f"Fetched datsa: (Train: {len(train_dataloader)}, Val: {len(validation_dataloader)})")
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=cfg.hyperparams.lr)

    # only needs to be called for local development
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))
    wandb.init(
        project = f"mlops-mnist-sign-language-{cfg.base.experiment_name}",
        entity = "mlops-mnist",
        config = {
            "learning_rate": cfg.hyperparams.lr,
            "epochs": cfg.hyperparams.epochs,
            "batch_size": cfg.data_fetch.batch_size,
            "architecture": "CNN",
            "dataset": "American-Sign-Language-MNIST",
        },
        mode=cfg.base.wandb_mode
    )

    logger.info("Start training")
    for epoch in range(cfg.hyperparams.epochs):
        running_train_loss = 0
        last_train_loss = 0
        validation_loss = 0

        # Start training step
        kwargs = {
            "cfg": cfg,
            "epoch": epoch,
            "logger": logger,
            "running_train_loss": running_train_loss,
        }
        model, optimizer, running_train_loss = train_step(model, train_dataloader, optimizer, criterion, **kwargs)
        
        # Start validation step
        accuracy, validation_loss = validate_step(model, validation_dataloader, criterion, validation_loss)

        # Done training
        logger.info(f'--- Epoch {epoch+1}/{cfg.hyperparams.epochs} | accuracy: {accuracy} | train loss: {last_train_loss} | val loss: {validation_loss / len(validation_dataloader)} ---\n')
        wandb.log({
            "accuracy" : accuracy,
            "validation_loss" : validation_loss / len(validation_dataloader)
        })

        # Check if bucket is mounted to image
        if os.path.isdir(f"/gcs/{cfg.data_fetch.gcp_bucket_name}"):
            torch.save(model.state_dict(), f"/gcs/{cfg.data_fetch.gcp_bucket_name}/models/model_{cfg.base.experiment_name}.pt") # saves to gcp
        # Saves when working locally, not in a container
        elif os.path.isdir("models/"): 
            torch.save(model.state_dict(), f"models/model_{cfg.base.experiment_name}.pt")

    logger.info("Done")
    wandb.finish()

if __name__ == '__main__':
    train()