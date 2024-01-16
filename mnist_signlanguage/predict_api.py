import datetime
import io
import os

import torch
from hydra import initialize, compose

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
        ])

    def __getitem__(self, idx):
        return (self.transform(self.images[idx]), 0)

    def __len__(self):
        return len(self.images)

@app.post("/predict/")
async def predict_images(images_files: list[UploadFile] = File(...)):

    with initialize(config_path="config", job_name="predict", version_base='1.3'):
        cfg = compose(config_name="train_model")

    start_time = datetime.datetime.now()

    images_byte = [f.file.read() for f in images_files]
    images_pil = [Image.open(io.BytesIO(b)) for b in images_byte]

    dataset = DummyDataset(images_pil)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)

    model = get_timm()
    if os.path.isdir(f"/gcs/{cfg.data_fetch.gcp_bucket_name}"):
        print("Loading weights for model for GS Bucket")
        model.load_state_dict(torch.load(f"/gcs/{cfg.data_fetch.gcp_bucket_name}/model_default-model.pt"))
    elif os.path.isdir(f"/models/model_default-model.pt"):
        print("Loading weights for model from local file")
        model.load_state_dict(torch.load(f"/gcs/{cfg.data_fetch.gcp_bucket_name}/model_default-model.pt"))
    
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