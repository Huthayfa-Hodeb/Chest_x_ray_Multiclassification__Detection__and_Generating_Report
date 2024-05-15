import gradio as gr
import cv2
import json
from ultralytics import RTDETR
import numpy as np
from transformers import VisionEncoderDecoderModel , ViTFeatureExtractor
from transformers import AutoTokenizer ,  GPT2Config
import torch


class config : 
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"


feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token

# Load the RTDETR model
model_det = RTDETR('model/best.pt')

model_report = VisionEncoderDecoderModel.from_pretrained('model/vit-gpt')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_report.to(device)

def predict(image):

    results = model_det([image])
    result = results[0]
    data = json.loads(result.tojson())

    if data:
        image_det = result.plot()
    
        report_text = tokenizer.decode(model_report.generate(feature_extractor(image, return_tensors="pt").pixel_values.to(device), temperature = 1, max_length = 100)[0])

        return image_det, report_text.replace("<|endoftext|>", "")

    else:
        cv2.rectangle(image, (0, image.shape[0] - 100), (300, image.shape[0]), (255, 255, 255), -1)
        cv2.putText(image, "Clean bill of health", (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        return image, "Clean bill of health"


# Create a Gradio interface
demo = gr.Interface(predict, 
                    inputs=gr.Image(), 
                    outputs=[gr.Image(label="Generated Image"), gr.Textbox(label="Report", lines=8)], 
                    title="Chest X-Ray Tumor Detection - Report Generation", 
                    theme="soft",
                    allow_flagging="never")

# Launch the interface
demo.launch()