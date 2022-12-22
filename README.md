# Project2 - CAP6610

Music Genre Binary Classifier implemented using Transformer models. Download the Non-Prog and Prog Training sets available on dropbox. Implement a prog-rock vs. the world
classifier on the training set. Document the mis-classification errors. This is a group project for Machine Learning Course.

## Table of Contents
- [Project2 - CAP6610](#project2---cap6610)
  - [Table of Contents](#table-of-contents)
  - [Installation](#installation)
  - [Project Structure](#project-structure)
  - [Reproduce Results](#reproduce-results)
  - [Dataset](#dataset)
  - [Algorithms](#algorithms)
    - [Support Vector Machine (SVM)](#support-vector-machine-svm)
    - [Random Forest](#random-forest)
    - [Least-Squares Classification](#least-squares-classification)
    - [Manifold Learning (UMAP)](#manifold-learning-umap)
  - [Next Steps](#next-steps)

<hr>

## Installation
Download the repository:
```console
git clone https://github.com/dani2442/Project2
```
Firstly, navigate to the folder and create a python environment.
```console
python -m venv venv
```
We activate the virutal environment. Depending on the OS
- For Linux:

    ```console
    source venv/bin/activate
    ```
- For Windows:

    ```console
    venv/Scripts/activate
    ```
Now, we install the necessary packages to run the algorithms such as torch, numpy, sklearn, etc:
```console
pip install -r requirements/requirements.txt
```
## Project Structure

## Reproduce Results

## Dataset


## Algorithms

### Support Vector Machine (SVM)

### Random Forest

### Least-Squares Classification


### Manifold Learning (UMAP)


## Next Steps
- **Transfer learning**: use a pretrained model in other task and fine-tuning for our classification task
- **Model Pre-training**: initialize the weights intelligently with some unsupervised technique. E.g., For each layer construct a symmectric counterpart and train it to reconstruct the original data.
- **Data augmentation/Synthetic data generation**: It is very common in computer vision task to implement some kind of data augmentation. It allows the model to generalize better/reduce overfitting, improve performance, make it more robust, etc.

    PyTorch has a very nice library [albumentations](https://albumentations.ai/docs/getting_started/image_augmentation/), that I have used previously with great results. It allows to apply multiple filters and transformations to images such as gaussian blur, change in brightness, cropping, flipping, etc.
    ```python
    transform = A.Compose([
        A.RandomCrop(width=256, height=256),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
    ])
    ``` 
    It will be very easy to implement a wrapper for the current dataset.
- **Add Layer Freezing options/configuration**: currently when dissecting the model, I freeze the rest of the model. However, it might be helpful to implement some kind of temporary warming or not freezing it completely.
