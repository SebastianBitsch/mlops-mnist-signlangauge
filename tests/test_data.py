from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torch
import os.path
import pytest


@pytest.mark.skipif(not os.path.exists('data/'), reason="Data files not found")
def test_data():
    # Load the MNIST training and test datasets
    test_dataset = torch.load(
        'data/processed/test/test_images.pt')
    train_dataset = torch.load(
        'data/processed/train/train_images.pt')

    # Assert the shape of each data point
    for image in DataLoader(train_dataset, batch_size=1):
        assert image.shape == torch.Size([1, 1, 28, 28]) or image.view(-1).shape == torch.Size([784]), \
            "Each data point in the training set should have shape [1, 28, 28] or [784]"

    for image in DataLoader(test_dataset, batch_size=1):
        assert image.shape == torch.Size([1, 1, 28, 28]) or image.view(-1).shape == torch.Size([784]), \
            "Each data point in the test set should have shape [1, 28, 28] or [784]"

    # Read labels from files
    with open('data/processed/train/train_labels.txt', 'r') as f:
        train_labels = set(int(label.strip()) for label in f.readlines())

    with open('data/processed/test/test_labels.txt', 'r') as f:
        test_labels = set(int(label.strip()) for label in f.readlines())

    # Assert that all labels are represented
    label_set = set([i for i in range(25)])
    label_set.remove(9)
    assert train_labels == label_set, "Not all labels are represented in the training set"
    assert test_labels == label_set, "Not all labels are represented in the test set"
