import torch



def to3channels(images : torch.Tensor) -> torch.Tensor:
    """
    Duplicates the 1 channel image to a 3 channeled image by concatenating it to itself 3 times
    
    ARGS: Tensor of shape (batch_size, 1, 28, 28)
    
    RETURNS: Tensor of shape (batch_size, 3, 28, 28)
    """
    images = images.unsqueeze(1)
    images = torch.cat([images, images, images], dim=1) # convert to 3 channels (RGB)
    return images