import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

import wandb
import logging

from models.model import Net



# Model Hyperparameters
dataset_path = "datasets"
cuda = True
DEVICE = torch.device("mps" if cuda else "cpu")
batch_size = 100
x_dim = 784
hidden_dim = 400
latent_dim = 20
lr = 1e-3
epochs = 20

train_dataset = ...
test_dataset = ...

train_dataloader = ...
test_dataloader = ...

model = Net().to(DEVICE)

loss_fn = nn.BCELoss()
optimizer = Adam(model.parameters(), lr=lr)

wandb.watch(model, log_freq=100)


for epoch in range(epochs):
    total_loss = 0
    model.train()

    # Train
    for batch_idx, (x, y) in enumerate(train_dataloader):

        y_pred = model(x)
        loss = loss_fn(y, y_pred)
        loss.backward()

        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch_idx % 100 == 0:
            wandb.log({"loss": loss})
            
    logging.info(
        "\tEpoch", epoch + 1, "complete!", 
        "\tAverage Loss: ", total_loss / (batch_idx * batch_size)
    )

    # Validate
    model.validate()
    for batch_idx, (x, _) in enumerate(test_dataloader):
        pass


print("Done")