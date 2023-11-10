# Deep-learning-project-4-Diabetic-retinopathy-detection-
Diabetic retinopathy is an eye disease, in which damage occurs to the retina due to diabetes. The longer a person has diabetes and the less controlled the blood sugar, the higher the chances of developing diabetic retinopathy. here. we provide the automatic methods for diabetic retinopathy detection.
TFRecords: 

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

