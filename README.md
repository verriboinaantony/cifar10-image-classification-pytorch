# 🚀 CIFAR-10 Image Classification using PyTorch

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![Status](https://img.shields.io/badge/Project-Completed-success)
![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-orange)

---

## 📌 Overview

This project presents a structured and research-oriented approach to image classification using deep learning on the CIFAR-10 dataset. Instead of directly applying complex architectures, the focus is on understanding how model design decisions affect performance.

Starting from a simple baseline, the project progressively improves model accuracy through systematic experimentation, architectural refinement, and training optimization.

---

## 🎯 Objectives

- Establish a baseline model for image classification
- Analyze limitations of fully connected networks on image data
- Develop and optimize Convolutional Neural Networks (CNNs)
- Apply regularization and data augmentation techniques
- Evaluate performance through controlled experiments
- Deploy the trained model for real-world inference

---

## 📂 Dataset

CIFAR-10 is a widely used benchmark dataset consisting of:

- **60,000 RGB images**
- **10 classes**: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Image size: **32 × 32 pixels**

---

## 🧠 Methodology

### 1. Baseline Model (MLP)

A Multi-Layer Perceptron (MLP) was implemented as a baseline by flattening images into vectors and passing them through fully connected layers.

**Limitation:**
The model fails to capture spatial dependencies between pixels, resulting in limited performance.

---

### 2. Convolutional Neural Network (CNN)

A CNN architecture was introduced to preserve spatial structure and extract local features such as edges and textures.

**Impact:**
Significant improvement in performance by leveraging convolutional operations.

---

### 3. Model Refinement

The CNN was enhanced using:

- **Batch Normalization** → stabilizes training
- **Dropout** → reduces overfitting
- **Deeper architecture** → improves representational capacity

---

### 4. Data Augmentation

To improve generalization, the following techniques were applied:

- Random horizontal flipping
- Random cropping

**Insight:**
Augmentation increases training difficulty initially, but improves performance with longer training.

---

## 📊 Results

| Model                                       | Accuracy |
| ------------------------------------------- | -------- |
| MLP (Baseline)                              | ~52%     |
| Basic CNN                                   | ~73%     |
| Improved CNN                                | ~77.5%   |
| CNN + Data Augmentation (extended training) | **82%+** |

---

## 📈 Key Insights

- Fully connected networks are not suitable for image data due to loss of spatial structure
- CNNs significantly improve performance by learning local patterns
- Regularization must be balanced with model complexity
- Data augmentation improves generalization but requires longer training
- Performance gains come from systematic improvements, not complexity alone

---

## ⚙️ Tech Stack

- Python
- PyTorch
- Torchvision
- Matplotlib
- Flask (for deployment)

---

## 🏗️ Project Structure

```bash
.
├── model.py
├── train.py
├── data_loader.py
├── predict.py
├── app.py
├── cnn_model.pth
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone repository

```bash
git clone https://github.com/<your-username>/cifar10-image-classification-pytorch.git
cd cifar10-image-classification-pytorch
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the model

```bash
python train.py
```

### 5. Run prediction

```bash
python predict.py
```

---

## 🌐 Deployment

The trained model is integrated into a Flask-based web application that allows users to upload images and receive predictions in real time.

---

## 🔮 Future Work

- Implement advanced architectures (ResNet, EfficientNet)
- Hyperparameter optimization
- Transfer learning for improved performance
- Cloud-based deployment
- Integration with MLOps tools (MLflow, Docker)

---

## 🧠 Conclusion

This project demonstrates that strong performance in deep learning can be achieved through careful experimentation, architectural understanding, and iterative refinement. The emphasis on analysis and controlled improvements makes the approach both practical and research-oriented.

---
