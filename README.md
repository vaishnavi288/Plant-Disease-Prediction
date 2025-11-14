# 🌿 Plant Disease Prediction using CNN

## 📘 Overview
This project focuses on building an intelligent system that predicts *plant diseases from leaf images* using *Convolutional Neural Networks (CNNs)*. Early detection of plant diseases can help farmers take timely actions, prevent crop loss, and improve yield.  
The model is trained on the *PlantVillage Dataset* available on *Kaggle*, which contains thousands of labeled images of healthy and diseased leaves across multiple crop types.

---

## 🎯 Objective
To develop a *deep learning model* capable of accurately identifying plant diseases from leaf images and classifying them into specific disease categories.

---

## 🧠 Key Features
- 📸 *Image-based disease detection* using CNN  
- 🔍 *Automatic classification* of healthy and diseased leaves  
- 🧩 *Preprocessing and data augmentation* for better accuracy  
- 📊 *Model evaluation* using accuracy, precision, recall, and F1 score  
- ☁ *Compatible with Google Colab / Jupyter Notebook*  
- 🖼 *Supports multiple plant species* (tomato, potato, maize, etc.)

---

## 🧰 Tools and Technologies
| Category | Tools / Libraries |
|-----------|-------------------|
| Programming Language | Python |
| Deep Learning Framework | TensorFlow, Keras |
| Image Processing | OpenCV, NumPy |
| Visualization | Matplotlib, Seaborn |
| Dataset Source | Kaggle (PlantVillage Dataset) |
| Environment | Google Colab / Jupyter Notebook |

---

## 📂 Dataset Description
*PlantVillage Dataset (Kaggle):*  
A large open-access dataset containing *50,000+ images* of healthy and diseased leaves from *38 different classes*.  
Each image belongs to a specific crop and disease type. Example categories include:
- Tomato — Healthy, Late Blight, Early Blight  
- Potato — Healthy, Late Blight, Early Blight  
- Apple — Healthy, Scab, Rust  
- Corn — Healthy, Northern Leaf Blight, Common Rust  

*Dataset Link:* [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)

---

## ⚙ System Workflow

1. *Data Collection:* Import images from the PlantVillage dataset.  
2. *Preprocessing:* Resize, normalize, and augment images to improve generalization.  
3. *Model Building:* Construct a CNN model using Keras Sequential API.  
4. *Training:* Train the model on the processed dataset.  
5. *Evaluation:* Measure model performance using accuracy metrics.  
6. *Prediction:* Use the trained model to classify new leaf images.

---

## 🧬 CNN Architecture (Example)
- *Input Layer:* 128×128 RGB image  
- *Convolution Layers:* Feature extraction using filters  
- *Pooling Layers:* Dimensionality reduction  
- *Flatten Layer:* Converts 2D to 1D feature vector  
- *Dense Layers:* Learn complex relationships  
- *Output Layer:* Softmax activation for multi-class classification

---

## 🚀 How to Run
Upload the file plant_disease_prediction.ipynb to Google Colab
trained model plant_disease_model.keras is stored in Google Drive.
* https://drive.google.com/drive/folders/1KfcC4xt511VuvdLTH0q-FHIqYAlvw3OG?usp=sharing
Load the Trained Model
Upload a Leaf Image for Prediction
Preprocess the Image and Predict the Disease
View Output
