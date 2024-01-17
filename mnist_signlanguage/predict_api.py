import datetime
import io
import os

import torch

from torch.utils.data import DataLoader, Dataset
from fastapi import FastAPI, File, UploadFile
from torchvision import transforms
from PIL import Image

from mnist_signlanguage.models.modelTIMM import get_timm
from mnist_signlanguage.predict_model import predict

# Build app
app = FastAPI()

class DummyDataset(Dataset):
    """TensorDataset with support of transforms"""
    def __init__(self, images: list):
        self.images = images
        self.transform = transforms.Compose([
            transforms.Resize(28), # 28 is too small for single images. idk why 33 is min
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            lambda x: x * 255      # scale to range 0-255 - images outputted from grayscale are in range 0-1
        ])

    def __getitem__(self, idx):
        return (self.transform(self.images[idx]), 0) # label is not used for prediction

    def __len__(self):
        return len(self.images)

@app.post("/predict/")
async def predict_images(images_files: list[UploadFile] = File(...)):

    start_time = datetime.datetime.now()

    images_byte = [f.file.read() for f in images_files]
    images_pil = [Image.open(io.BytesIO(b)) for b in images_byte]

    dataset = DummyDataset(images_pil)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    model = get_timm()
    if os.path.isdir(f"/gcs"):
        print("Loading weights for model for GS Bucket")
        model.load_state_dict(torch.load(f"/gcs/models/model_default-model.pt"))
    elif os.path.isdir("models"):
        print("Loading weights for model from local file")
        model.load_state_dict(torch.load(f"models/model_default-model.pt"))
    else:
        print("Error: Failed to load weights, using untrained network - results will be poor")
    
    model.eval()

    with torch.no_grad():
        pred = predict(model, dataloader).argmax(dim=1)

    # bookkeeping for fun    
    end_time = datetime.datetime.now()
    time_diff = (end_time - start_time)

    return {
        "inference_time": f'{round(time_diff.total_seconds() * 1000)} ms',
        "predictions": {
            "class_id": pred.tolist(),
        }
    }