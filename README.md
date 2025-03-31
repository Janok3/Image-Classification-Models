# Comparative Analysis of Image Classification Algorithms: A Study of CNN, Logistic Regression, K-means, GAN, and SNN


## Description

This project provides an experimental framework for evaluating different types of artifical intelligence models on cats and dogs image classification task. The implemented models are:

* **Logistic Regression** (Supervised ML)
* **K-Means Clustering** (Unsupervised ML)
* **Convolutional Neural Network** (Supervised DL)
* **GAN Discriminator** (Unsupervised DL)
* **Spiking Neural Network**

## Getting Started

### Dependencies

The project relies on several popular libraries. Make sure you have the following installed:

* Python 3.7.4
* NumPy (1.21.6)
* Matplotlib (3.5.3)
* Pillow (9.5.0)
* scikit-learn (1.0.2)
* TensorFlow (2.10.1)
* Torch (1.13.1)
* TorchVision (0.14.1)
* nengo (3.2.0)
* nengo-dl (3.6.0)

### Installing

Install these dependencies using pip
```
pip install numpy==1.21.6 matplotlib==3.5.3 pillow==9.5.0 scikit-learn==1.0.2 tensorflow==2.10.1 torch==1.13.1 torchvision==0.14.1 nengo==3.2.0 nengo-dl==3.6.0
```

### Usage

1. Prepare dataset
Organize your dataset into the following format:
```
dataset/
  ├── train/
  │    ├── class1/
  │    ├── class2/ 
  └── test/
       ├── class1/
       ├── class2/
      
```
2. Run the script
```
pyhton main.py
```

## Authors

Contributors names and contact info
- Çağan Çakır
- Janok N. Dinçer
