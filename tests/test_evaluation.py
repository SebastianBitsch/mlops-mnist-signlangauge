from mnist_signlanguage.utils.evaluation import evaluate, Evaluation_Metrics
from mnist_signlanguage.train_model import instantiate_training_objects
from tests import CONFIG
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from pytest import approx

def test_evaluate():
    device_name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    DEVICE = torch.device(device_name)
    model, train_dataloader, validation_dataloader, criterion, optimizer = instantiate_training_objects(CONFIG, DEVICE)
    
    evaluate(model, validation_dataloader, criterion, CONFIG, NUM_CLASSES = 25)
    

def test_metrics():
    NUM_CLASSES = 25
    metrics = Evaluation_Metrics()
    
    torch.manual_seed(0)
    
    y_pred = torch.randint(0, NUM_CLASSES, (500,))
    y_true = torch.randint(0, NUM_CLASSES, (500,))
    
    # for i in range(5):
    #     yp = y_pred[i*100:(i+1)*100]
    #     y_true_ = y_true[i*100:(i+1)*100]
    #     metrics.update(yp, y_true_)
    
    metrics.update(y_pred, y_true)
    
    torch_scores = metrics.compute()
    
    acc = torch_scores["test_accuracy"]
    prec = torch_scores["test_precision"]
    rec = torch_scores["test_recall"]
    f1 = torch_scores["test_f1"]
    
    
    assert acc == approx(accuracy_score(y_true, y_pred)), f"Expected {acc}, got {accuracy_score(y_true, y_pred)}"
    assert prec == approx(precision_score(y_true, y_pred, average='macro')), f"Expected {prec}, got {precision_score(y_true, y_pred, average='macro')}"
    assert rec == approx(recall_score(y_true, y_pred, average='macro')), f"Expected {rec}, got {recall_score(y_true, y_pred, average='macro')}"
    assert f1 == approx(f1_score(y_true, y_pred, average='macro')), f"Expected {f1}, got {f1_score(y_true, y_pred, average='macro')}"
    

        
    
    
    