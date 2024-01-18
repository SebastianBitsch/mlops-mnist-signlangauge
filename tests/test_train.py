from tests import CONFIG
import torch
from mnist_signlanguage.train_model import instantiate_training_objects, train_batch, train_step, validate_step, train_loop


def test_instantiate_training_objects():
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    model, train_dataloader, validation_dataloader, criterion, optimizer = instantiate_training_objects(CONFIG, DEVICE)
    
    # Check that the model is a torch model
    assert isinstance(model, torch.nn.Module)
    
    # Check that the dataloaders are torch dataloaders
    assert isinstance(train_dataloader, torch.utils.data.DataLoader)
    assert isinstance(validation_dataloader, torch.utils.data.DataLoader)
    
    # Check that the criterion is a torch criterion
    assert isinstance(criterion, torch.nn.modules.loss._Loss)
    
    # Check that the optimizer is a torch optimizer
    assert isinstance(optimizer, torch.optim.Optimizer)

def test_train_batch():
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    model, train_dataloader, validation_dataloader, criterion, optimizer = instantiate_training_objects(CONFIG, DEVICE)
    
    data_batch = next(iter(train_dataloader))
    running_train_loss = 0
    
    model2, optimizer2, criterion2, running_train_loss2, train_loss2 = train_batch(model, data_batch, optimizer, criterion, running_train_loss)
    
    # Check that the parameters of the model have been updated
    assert model.parameters() != model2.parameters()    
    
    # Check that the running_train_loss has been updated
    assert running_train_loss != running_train_loss2
    

def test_train_step():
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    model, train_dataloader, validation_dataloader, criterion, optimizer = instantiate_training_objects(CONFIG, DEVICE)
    
    # Get subset of train_dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataloader.dataset[:10], batch_size=2)
        
    model2, optimizer2, running_train_loss = train_step(model, train_dataloader, optimizer, criterion, CONFIG)
    
    # Check that the parameters of the model have been updated
    assert model.parameters() != model2.parameters()
    assert running_train_loss != 0
    

def test_validate_step():
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    model, train_dataloader, validation_dataloader, criterion, optimizer = instantiate_training_objects(CONFIG, DEVICE)
    
    accuracy, validation_loss = validate_step(model, validation_dataloader, criterion, 0)
    
    assert type(accuracy) == float
    assert type(validation_loss) == torch.Tensor

def test_train_loop():
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    model, train_dataloader, validation_dataloader, criterion, optimizer = instantiate_training_objects(CONFIG, DEVICE)
    
    # Get subset of train_dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataloader.dataset[:10], batch_size=2)
    
    model = train_loop(model, train_dataloader, validation_dataloader, criterion, optimizer, CONFIG)
    


    



