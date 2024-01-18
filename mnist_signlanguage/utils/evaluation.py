import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils.logging import with_default_logging
from utils.operations import to3channels

# class Evaluation_Metrics():
#     """
#     Class for evaluating the model
#     """
    
#     def __init__(self, num_classes) -> None:
#         self.accuracy = torch_metrics.MulticlassAccuracy(num_classes=num_classes)
#         self.precision = torch_metrics.MulticlassPrecision(num_classes=num_classes, average="macro")
        
        
#         # Recall Function is straight up bugged
#         # https://github.com/pytorch/torcheval/issues/150
#         # https://github.com/pytorch/torcheval/pull/166
        
#         # self.recall = torch_metrics.MulticlassRecall(num_classes=num_classes, average="macro") 
        
#         self.f1 = torch_metrics.MulticlassF1Score(num_classes=num_classes, average="macro")
#         # self.precision_recall_curve = torch_metrics.MulticlassPrecisionRecallCurve(num_classes=num_classes)
#         # self.confusion_matrix = torch_metrics.MulticlassConfusionMatrix(num_classes=num_classes)
    
#     def update(self, y_pred, y_true):
#         """
#         Update metrics on the model
        
#         ARGS:
#             y_pred: PyTorch tensor
#             y_true: PyTorch tensor
        
#         RETURNS:
#             None
#         """
#         self.accuracy.update(y_pred, y_true)
#         self.precision.update(y_pred, y_true)
#         # self.recall.update(y_pred, y_true)
#         self.f1.update(y_pred, y_true)
#         # self.precision_recall_curve.update(y_pred, y_true)
#         # self.confusion_matrix.update(y_pred, y_true)
        
#     def compute(self):
#         """
#         Compute metrics on the model
        
#         ARGS:
#             None
        
#         RETURNS:
#             dict
#         """
#         return {
#             "test_accuracy": self.accuracy.compute(),
#             "test_precision": self.precision.compute(),
#             # "test_recall": self.recall.compute(),
#             "test_f1": self.f1.compute(),
#             # "test_precision_recall_curve": self.precision_recall_curve.compute(),
#             # "test_confusion_matrix": self.confusion_matrix.compute()
#         }

    
class Evaluation_Metrics():
    """
    Class for evaluating the model
    """
    
    def __init__(self) -> None:
        self.y_pred = torch.tensor([])
        self.y_true = torch.tensor([])
    
    def update(self, y_pred, y_true):
        """
        Update metrics on the model
        
        ARGS:
            y_pred: PyTorch tensor
            y_true: PyTorch tensor
        
        RETURNS:
            None
        """
        self.y_pred = torch.cat((self.y_pred, y_pred), 0)
        self.y_true = torch.cat((self.y_true, y_true), 0)
        
    def compute(self):
        """
        Compute metrics on the model
        
        ARGS:
            None
        
        RETURNS:
            dict
        """
        return {
            "test_accuracy": accuracy_score(self.y_true, self.y_pred),
            "test_precision": precision_score(self.y_true, self.y_pred, average='macro'),
            "test_recall": recall_score(self.y_true, self.y_pred, average='macro'),
            "test_f1": f1_score(self.y_true, self.y_pred, average='macro'),
            # "test_confusion_matrix": confusion_matrix(self.y_true, self.y_pred)
        }

@with_default_logging("Start evaluation")
def evaluate(model, test_dataloader, criterion, cfg, NUM_CLASSES = 25):
    
    metrics = Evaluation_Metrics()
    
    
    with torch.no_grad():
        model.eval()
        

        

        
        for batch_idx, (images, labels) in enumerate(test_dataloader):
            images = to3channels(images)
            
            preds = model(images)
            
            y_pred = torch.argmax(preds, dim=1)
            
            metrics.update(y_pred, labels)
            
    evaluation_scores = metrics.compute()
    
    wandb_data = evaluation_scores
    log_msg = "\n".join([
    f"\n--- SCORING METRICS ON TEST DATA ---",
    f"--- Accuracy: {evaluation_scores['test_accuracy']} ---",
    f"--- Precision: {evaluation_scores['test_precision']} ---",
    f"--- Recall: {evaluation_scores['test_recall']} ---",
    f"--- F1: {evaluation_scores['test_f1']} ---",
    ])
    
    return (evaluation_scores), (log_msg, wandb_data)