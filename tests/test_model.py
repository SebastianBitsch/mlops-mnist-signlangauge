import torch

from mnist_signlanguage.models.modelTIMM import get_timm 

def test_model():
    in_features = 1
    n_classes = 25
    model = get_timm()
    x = torch.randn(16, 3, 28, 28)

    out = model(x)
    assert out.shape == (16, n_classes)

