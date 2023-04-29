# Image classification with ViT using CIFAR-10 Dataset

## Team: 
Piotr Piesiak, Witold Płecha, Maurycy Borkowski

## Overview

This project is designed to classify images from the CIFAR-10 dataset. We fine-tuned pretrained Vision Transformer, which was created by Google Brain. To keep a well-organized project structure we used Kedro framework. We implemented pipelines to make the workflow easier.

## ViT by Google Brain
* The description below was taken from the Transformers-Tutorial with ViT: https://github.com/NielsRogge/Transformers-Tutorials


The Vision Transformer (ViT) is basically BERT, but applied to images. It attains excellent results compared to state-of-the-art convolutional networks. In order to provide images to the model, each image is split into a sequence of fixed-size patches (typically of resolution 16x16 or 32x32), which are linearly embedded. One also adds a [CLS] token at the beginning of the sequence in order to classify images. Next, one adds absolute position embeddings and provides this sequence to the Transformer encoder.
![Alt text](./img/Vit.png?raw=true "Title")
* Paper: https://arxiv.org/abs/2010.11929
* Official repo (in JAX): https://github.com/google-research/vision_transformer

## Dataset overview
We will collect data from Hugging Face dataset cifar-10 [🤗 CIFAR-10](https://huggingface.co/datasets/cifar10). This dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
* Class labels:
airplane, frog, bird, horse, automobile, deer, dog, cat, truck, ship

## Instalation and usage

Dependencies are declared in `src/requirements.txt` 

To install them, run:

```
pip install -r src/requirements.txt
```

To run all pipelines:
```
kedro run
```

## Download the dataset

The data can be downloaded from Hugging Face manually from: [🤗 CIFAR-10](https://huggingface.co/datasets/cifar10) (remember to put them into [02_intermediate](data/02_intermediate) folder). One can also do this using command:
```
kedro run --pipeline=download_data
```
The data will be downloaded into the [02_intermediate](data/02_intermediate) folder. One can change parameters in [download_params](conf/base/parameters/download_data.yml) file in order to modify train, validation and test sizes. The model performs very well on realtively small amount of data: 5000 samples in train and validation sets (4500 train, 500 val) and 2000 samples in test set. Such parameters will speed up the training process.

## Preprocessing the dataset

We use HuggingFace Datasets' set_transform method, which performs data augmentation on-the-fly. This method transforms data only when given example is accessed. That's why there is not separete pipeline for preprocessing. The transform is applied during training and evaluation. One can find implementation of data augmentation here [preprocessing](src/image_classification_with_vit/pipelines/train_model/processing_nodes.py).

## Train the model

To train the model use below command:
```
kedro run --pipeline=train_model
```
We fine-tune pretrained model, so the number of training parameters is small. You can change them here [trainig_parameters](conf/base/parameters/train_model.yml). After trainig, the model will be saved into the [06_model](data/06_model) folder. Our fine-tuned model is uploaded on this repository ([06_model](data/06_model)) and ready to use. You can change the device (cpu / cuda) in the [parameters](conf/base/parameters.yml) file.

## Evaluate the model

To evaluate the model use below command:
```
kedro run --pipeline=evaluate_model
```
After evaluation the confusion matrix can be found in the [08_reporting](data/08_reporting) folder. We present here the results of testing our fine-tuned model:

![Alt text](./data/08_reporting/confusion_matrix.png?raw=true "Title")

We uploaded fine-tuned model, so one can run this command without training the model.

## Weights&Biases report
Here you can find the report 

sample image
