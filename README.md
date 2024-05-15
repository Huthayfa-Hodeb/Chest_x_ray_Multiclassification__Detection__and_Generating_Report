# Chest x-ray Multiclassification, Detection and Generating_Report

This is the implementation of RT-DETR model and VIT-GPT2

# Requirements

# Download RT-DETR model Or VIT-GPT2

You can download the models we trained for each dataset from the following links:
- [RT-DETR](https://drive.google.com/file/d/1LtnZ52JKhEuOYGLxhamAxKNVftV1N8xe/view?usp=sharing)
- [VIT-GPT2](https://drive.google.com/file/d/14ooNq_5hDDvNlPTJtqMW06AbRr9Hc3fR/view?usp=sharing)

# Datasets

We use two datasets (NIH and MIMIC-CXR) in our project.

For `NIH`, you can download the dataset from [here](https://drive.google.com/file/d/14ooNq_5hDDvNlPTJtqMW06AbRr9Hc3fR/view?usp=sharing)

For `NIH`, in `RT-DETR model` format you can download the dataset from [here](https://drive.google.com/file/d/1LtMebJa8SWne_0d7cAV8Uyg2nQdgZk-q/view?usp=sharing)

For `MIMIC-CXR`, you can download the dataset from [here](https://huggingface.co/datasets/hongrui/mimic_chest_xray_v_1)

# Code Files
- [ChestXRay](./ChestXRay.ipynb) : in this file we read the data, extract the bounding box images from all images, preprocess them, and create a CSV file in Roboflow format.
- 
