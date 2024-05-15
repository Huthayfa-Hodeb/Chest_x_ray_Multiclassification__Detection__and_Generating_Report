# Chest x-ray Multiclassification, Detection and Generating_Report

This is the implementation of RT-DETR model and VIT-GPT2.

# Requirements
- `gradio==4.31.2`
- `opencv-python==4.8.0.74`
- `transformers==4.40.2`
- `torch==2.2.2`
- `torchvision==0.17.2`
- `ultralytics==8.2.2`
- `roboflow`

# Download RT-DETR model Or VIT-GPT2

You can download the models we trained for each dataset from the following links :
- [RT-DETR](https://drive.google.com/file/d/1LtnZ52JKhEuOYGLxhamAxKNVftV1N8xe/view?usp=sharing).
- [VIT-GPT2](https://drive.google.com/file/d/14ooNq_5hDDvNlPTJtqMW06AbRr9Hc3fR/view?usp=sharing).

# Datasets

We use two datasets (NIH and MIMIC-CXR) in our project.

For `NIH`, you can download the dataset from [here](https://drive.google.com/file/d/14ooNq_5hDDvNlPTJtqMW06AbRr9Hc3fR/view?usp=sharing).

For `NIH`, in `RT-DETR model` format you can download the dataset from [here](https://drive.google.com/file/d/1LtMebJa8SWne_0d7cAV8Uyg2nQdgZk-q/view?usp=sharing).

For `MIMIC-CXR`, you can download the dataset from [here](https://huggingface.co/datasets/hongrui/mimic_chest_xray_v_1).

# Code Files
- [ChestXRay](./ChestXRay.ipynb) : in this file we read the data, extract the bounding box images from all images, preprocess them, and create a CSV file in Roboflow format.
- [chest detection model](./chest-detection-model.ipynb) : in this file we create the tumor detection model (`RT-DETR`).
- [final model](./final_model.ipynb) : in this file you can use `RT-DETR model` to make inferences.
- [image captioning](./image-captioning.ipynb) : in this file we create the report generating model (`VIT-GPT2`).
- [app](./app.py) : in this file you can run the gradio app and use two models to make inferences.
  
  Note: Before run app.py make sure to download two models and put theme in [model](./model) folder.
