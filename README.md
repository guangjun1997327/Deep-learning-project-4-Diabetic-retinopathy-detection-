# Deep Learning Project for Diabetic Retinopathy Detection

Diabetic retinopathy is a serious eye condition associated with diabetes, leading to damage in the retina. This project aims to provide automatic methods for the detection of diabetic retinopathy using deep learning techniques.

## Project Overview
This repository contains code for training and evaluating deep learning models to classify and detect diabetic retinopathy from retinal images. We use the IDRID (Indian Diabetic Retinopathy Image Dataset) and implement various models like VGG, ResNet, and a transfer learning approach for this purpose.

## Getting Started
### Prerequisites
- Python 3
- TensorFlow
- Other necessary libraries and dependencies (listed in `requirements.txt`)

### Preparing the Dataset
Convert the IDRID dataset into TFRecords for efficient input pipeline processing:

- under the folder `./input_pipeline/`, run the following command to convert IDRID dataset into TFRecords
`python3 create_smaller_tfrecord.py`. 

## Diabetic Retinopathy

- Binary classification:
  
  Training:
  - train vgg model:
  
    `python3 main.py --mode train --model_name vgg --num_classes 2`
  
  - train resnet model:
    
    `python3 main.py --mode train --model_name resnet --num_classes 2`
  - train transfer model:
  
    `python3 main.py --mode train --model_name transfer --num_classes 2`
  
  Test:
  - evaluate vgg model:
  
    `python3 main.py --mode evaluate --model_name vgg --num_classes 2`
  
  - evaluate resnet model:
  
    `python3 main.py --mode evaluate --model_name resnet --num_classes 2`
  
  - evaluate transfer model:
  
    `python3 main.py --mode evaluate --model_name transfer --num_classes 2`
  
  - evaluate ensemble model:
  
    `python3 ensemble.py --num_classes 2`
  
  Deep visualization:
  - visualization for vgg model:
  
    `python3 deep_visualization.py --model_name vgg --num_classes 2`
  
    visualization images are located under the folder: `diabetic_retinopathy/checkpoint/vgg/visualization/`
  
  - visualization for resnet model:
  
    `python3 deep_visualization.py --model_name resnet --num_classes 2`
  
    visualization images are located under the folder: `diabetic_retinopathy/checkpoint/resnet/visualization/`


- Multi-classes classification:

  Training:
  
    `python3 main.py --mode train --model_name transfer --num_classes 5`

  Test:

    `python3 main.py --mode evaluate --model_name transfer --num_classes 5`
