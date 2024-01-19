from mnist_signlanguage.utils.operations import to3channels
import torch

def test_to3channels():
    x = torch.randn(16, 28, 28)
    out = to3channels(x)
    assert out.shape == (16, 3, 28, 28)