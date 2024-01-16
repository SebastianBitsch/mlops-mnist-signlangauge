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

    model = Net().to(DEVICE)
    
    train_dataloader, validation_dataloader = fetch_dataloader(cfg.data_fetch, DEVICE)
    logger.info(f"Fetched data: (Train: {len(train_dataloader)}, Val: {len(validation_dataloader)})")
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

        # Start training loop
        model.train()
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            preds = model(images)

            loss = criterion(preds, labels)
            loss.backward()
            running_train_loss += loss.item()

            optimizer.step()

            if batch_idx % cfg.base.log_interval == cfg.base.log_interval - 1:
                last_train_loss = running_train_loss / cfg.base.log_interval
                running_train_loss = 0
                
                # Log to wandb
                logger.info(f"epoch: {epoch+1}/{cfg.hyperparams.epochs} | batch: {batch_idx+1}/{len(train_dataloader)} | loss: {last_train_loss}")
                wandb.log({"train_loss": last_train_loss})
        
        # Start validation
        model.eval()
        with torch.no_grad():
            correct_predictions = 0
            total_samples = 0
            for batch_idx, (images, labels) in enumerate(validation_dataloader):                
                
                preds = model(images)
                
                loss = criterion(preds, labels)
                validation_loss += loss
                
                correct_predictions += torch.sum(preds.argmax(dim=1) == labels).item()
                total_samples += len(labels)

        accuracy = correct_predictions / total_samples

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