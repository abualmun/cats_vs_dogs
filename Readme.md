# Cats vs Dogs Classifier 🐱🐶

![Output](screenshots/0.jpg)




This project implements a deep learning model using PyTorch to classify images of cats and dogs. The model is based on a pre-trained ResNet-18 architecture, fine-tuned on the popular **Cats vs Dogs** dataset.

## 📁 Dataset

- The dataset used is the **Dogs vs. Cats** dataset available on [Kaggle](https://www.kaggle.com/c/dogs-vs-cats/data).
- Images are organized into subfolders under `./petImages/` for use with `torchvision.datasets.ImageFolder`.

## 🧠 Model

- Base architecture: **ResNet-18** pretrained on ImageNet.
- Final fully connected layer adjusted to output 2 classes (Cat or Dog).
- Training is done using **CrossEntropyLoss** and **Adam optimizer**.

## 🏋️‍♂️ Training

Training is done in `main.ipynb` using the following setup:

- Images are resized to 224x224 and normalized.
- 80% of the dataset is used for training, 20% for validation.
- Batch size: 100
- Number of epochs: 10

After training, the model is evaluated for validation accuracy.

## 🧪 Testing

The `test.py` script takes an image path as an argument and outputs the prediction (Cat or Dog) along with a visual display.

**Usage:**
```bash
python test.py path_to_image.jpg
````

![Output](screenshots/1.jpg)


## 🗃️ Saved Model

Trained weights are saved to:

```
cats_vs_dogs_resnet18.pth
```

## 🛠️ Requirements

* Python 3.x
* torch
* torchvision
* matplotlib
* pillow

Install dependencies with:

```bash
pip install torch torchvision matplotlib pillow
```

## 🔍 Notes

* Normalization stats (`mean`, `std`) were calculated from the dataset manually (see commented code in `main.ipynb`).
* This project is intended for educational/demo purposes.

## 📦 Acknowledgments

* Dataset sourced from [Kaggle Dogs vs. Cats](https://www.kaggle.com/c/dogs-vs-cats).
* Built using PyTorch and torchvision.
