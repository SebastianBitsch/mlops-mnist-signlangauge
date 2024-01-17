from torch.utils.data import DataLoader
import torch
import os.path
import pytest


@pytest.mark.skipif(not os.path.exists("data/"), reason="Data files not found")
def test_data():
    # Load the MNIST training and test datasets
    test_dataset = torch.load("data/processed/test/test_images.pt")
    train_dataset = torch.load("data/processed/train/train_images.pt")

    # Assert the shape of each data point
    for image in DataLoader(train_dataset, batch_size=1):
        assert image.shape == torch.Size([1, 1, 28, 28]) or image.view(
            -1
        ).shape == torch.Size(
            [784]
        ), "Each data point in the training set should have shape [1, 28, 28] or [784]"

    for image in DataLoader(test_dataset, batch_size=1):
        assert image.shape == torch.Size([1, 1, 28, 28]) or image.view(
            -1
        ).shape == torch.Size(
            [784]
        ), "Each data point in the test set should have shape [1, 28, 28] or [784]"

    # Read labels from files
    with open("data/processed/train/train_labels.txt", "r") as f:
        train_labels = set(int(label.strip()) for label in f.readlines())

    with open("data/processed/test/test_labels.txt", "r") as f:
        test_labels = set(int(label.strip()) for label in f.readlines())

    # Assert that all labels are represented
    label_set = set([i for i in range(25)])
    label_set.remove(9)
    assert (
        train_labels == label_set
    ), "Not all labels are represented in the training set"
    assert test_labels == label_set, "Not all labels are represented in the test set"


def test_fetch_data():
    from mnist_signlanguage.data.make_dataset import fetch_dataloader

    device_name = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    DEVICE = torch.device(device_name)

    cfg = {
        "train_data_path": "data/processed/train/train_images.pt",
        "train_labels_path": "data/processed/train/train_labels.txt",
        "test_data_path": "data/processed/test/test_images.pt",
        "test_labels_path": "data/processed/test/test_labels.txt",
        # the hyperparameter for the data loader
        "batch_size": 1,
    }

    trainloader, testloader = fetch_dataloader(DEVICE, **cfg)

    # Assert that the dataloaders are not empty
    assert len(trainloader) > 0, "The training dataloader is empty"
    assert len(testloader) > 0, "The test dataloader is empty"

    # Assert the shape of each data point
    for image, label in trainloader:
        "Each data point in the training set should have shape [1, 28, 28]"
        assert image.shape == torch.Size([cfg["batch_size"], 28, 28])

        "Each label in the training set should have shape [1]"
        assert label.shape == torch.Size([1])
