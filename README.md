# Land Use Classification using Geospatial Deep Learning

Author: Akshay Raj M  
Program: M.Tech in Data Science and Artificial Intelligence  
Institution: Indian Institute of Technology, Tirupati

## Project Overview

This project develops a deep learning pipeline to classify land use categories from satellite imagery over the Delhi-NCR region. The objective is to identify major land use types such as cropland, vegetation, and built-up areas using satellite image patches and geospatial land cover data.

The system integrates geospatial processing with deep learning by using ESA WorldCover 2021 raster data to generate labels and Sentinel-2 RGB satellite imagery for model training. A pretrained ResNet18 model is fine-tuned using transfer learning to perform multi-class land use classification.

## Dataset

Satellite imagery patches are provided for the Delhi NCR region. Each image represents a 128 × 128 pixel RGB patch corresponding to a geographic coordinate.

Land cover labels are derived from the ESA WorldCover 2021 dataset (10 m resolution). For each image location, a corresponding raster patch is extracted and the dominant land cover class is used as the label.

To address class imbalance and simplify the classification problem, the original ESA classes were mapped into three categories:

| ESA Class Codes | Final Class |
|-----------------|-------------|
| 10, 20, 30      | Vegetation  |
| 40              | Cropland    |
| 50              | Built-up    |

Rare classes such as water and wetlands were excluded due to insufficient samples.

## Methodology

The workflow of the project consists of the following steps:

1. Load geospatial datasets including GeoJSON boundary files and ESA WorldCover raster data.
2. Extract raster patches corresponding to each satellite image coordinate.
3. Determine the dominant land cover class within each raster patch.
4. Construct labeled datasets and perform stratified train-test splitting.
5. Apply data augmentation techniques such as random rotation and horizontal flipping.
6. Train a deep learning model using transfer learning.
7. Evaluate model performance using accuracy and macro F1-score.

## Model Architecture

The classification model is based on the ResNet18 architecture pretrained on ImageNet.

Key modifications include:
- Freezing early convolutional layers to retain pretrained features
- Fine-tuning the final convolutional block
- Replacing the final fully connected layer to output three classes

Training configuration:
- Optimizer: Adam
- Learning rate: 1e-4
- Batch size: 32
- Loss function: Weighted Cross Entropy Loss
- Number of epochs: 10

Class weights were used to handle class imbalance.

## Results

Final model performance on the test dataset:

Accuracy: 92.8%  
Macro F1-score: 0.8797

Training and evaluation curves are available in the outputs directory.

## Repository Structure

```
AI_Sustainability/
│
├── data/
     └──raw/
│  
│
├── src/
│   ├── dataset.py
│   ├── model.py
│   └── train.py
│
├── outputs/
│   └── training_curves.png
│
├── main.py
├── Sustainability.pdf
└── README.md
```

## How to Run

1. Install required dependencies.

```
pip install -r requirements.txt
```

2. Run the training script.

```
python main.py
```

The script performs preprocessing, model training, evaluation, and saves the trained model and plots.

## Notes

The raw dataset and raster files are not included in the repository due to size limitations. They must be placed in the `data/raw/` directory before running the project.

## Acknowledgements

ESA WorldCover 2021 dataset was used as the source of land cover labels.  
Additional references and documentation were consulted for geospatial processing and deep learning implementation.