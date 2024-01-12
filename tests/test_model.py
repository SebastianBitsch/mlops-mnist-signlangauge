import torch

from mnist_signlanguage.models.model import Net 

def test_model():
    in_features = 1
    n_classes = 24
    model = Net(in_features = in_features, n_classes = n_classes)
    x = torch.randn(16, 1, 28, 28)

    out = model(x)
    assert out.shape == (16, n_classes)
    