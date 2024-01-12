from fastapi import FastAPI, File, UploadFile
from http import HTTPStatus
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
from typing import List

app = FastAPI()

model = VisionEncoderDecoderModel.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained(
    "nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

gen_kwargs = {"max_length": 16, "num_beams": 8, "num_return_sequences": 1}


def predict_step(images):  # Should be replaced with our own models prediction
    images_data = []
    for image_data in images:
        i_image = Image.open(image_data.file)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images_data.append(i_image)
    pixel_values = feature_extractor(
        images=images_data, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


@app.post("/cv_model/")
async def cv_model(images: List[UploadFile] = File(...)):
    predictions = predict_step(images)

    response = {
        "input_images": [image.filename for image in images],
        "predictions": predictions,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    return response
