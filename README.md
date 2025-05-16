# 🚗 Car Brands Classification using Custom CNN

This project focuses on classifying car images into different brands using a custom Convolutional Neural Network (CNN) model built from scratch. The goal is to achieve high accuracy in identifying car brands based on visual features.

---

## 📊 Project Overview

- **Task**: Multi-class image classification
- **Dataset**: Images of cars belonging to various brands (e.g., BMW, Audi, Mercedes, Toyota, etc.)
- **Model**: Custom CNN architecture (without using pre-trained models)
- **Framework**: PyTorch
- **Evaluation Metrics**: Accuracy, Loss, Confusion Matrix, Grad-CAM Visualizations

---

## 🗂️ Dataset Samples

Below are some random samples from the dataset with corresponding labels:

![Dataset Samples](path/to/dataset_samples.png)

---

## 🏗️ CNN Model Architecture

The custom CNN model consists of several convolutional and pooling layers, followed by fully connected layers for classification. ReLU activations and batch normalization are used to improve performance.

---

## 📈 Training & Validation Curves

Here are the learning curves showing **Accuracy** and **Loss** over training epochs:

### Accuracy Curve
![Accuracy Curve](path/to/accuracy_curve.png)

### Loss Curve
![Loss Curve](path/to/loss_curve.png)

---

## 🧠 Grad-CAM Visualization

Grad-CAM is used to highlight the important regions in the image that contributed to the classification decision.

![Grad-CAM Visualization](path/to/gradcam_visualization.png)

---

## 🧮 Confusion Matrix

The confusion matrix below illustrates the model's performance across different car brand classes.

![Confusion Matrix](path/to/confusion_matrix.png)

---

## ✅ Results Summary

| Metric        | Value |
|---------------|--------|
| Train Accuracy| XX %   |
| Val Accuracy  | XX %   |
| Test Accuracy | XX %   |

---

## 🚀 How to Run

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/car-brand-classification.git
   cd car-brand-classification
````

2. Install dependencies:

   ```
   pip install -r requirements.txt
   ```
3. Train and test the model:

   ```
   python main.py
   ```



