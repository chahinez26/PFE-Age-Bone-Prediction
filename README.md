# Bone Age Prediction Using Deep Learning

A deep learning-based system for automated bone age assessment from hand radiographs, designed to assist pediatricians and endocrinologists in clinical practice.

## Overview

This project develops a convolutional neural network (CNN) model that automatically predicts bone age from left-hand X-ray images. The model incorporates clinical information such as patient gender to enhance prediction accuracy and has been trained on both international and local Algerian datasets.

## Key Features

- **Automated Bone Age Prediction**: Deep learning model for rapid and accurate bone age estimation
- **Multi-Input Architecture**: Combines radiographic images with patient gender for improved accuracy
- **Clinical Integration**: Simple GUI application for easy use by healthcare professionals
- **Transfer Learning**: Custom fine-tuning approach adapted to local population characteristics
- **Performance**: Achieves MAE of 10.38 months on RSNA dataset and 12.86 months on local Bainem dataset


## Model Architecture

### Custom CNN Model

The model features a dual-input architecture:

- **Image Branch**: 4 convolutional blocks (32 → 64 → 128 → 256 filters) with batch normalization and max pooling
- **Gender Branch**: Dense layer processing binary gender input
- **Fusion Layer**: Concatenates image features with gender information
- **Output**: Single neuron predicting bone age in months

### Key Components

- **Input Size**: 224×224 grayscale images
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Regularization**: Dropout (0.3-0.5), Batch Normalization, Early Stopping

### Model Selection Process

To identify the optimal architecture, we conducted a comprehensive comparative study of four different approaches:

1. **Custom CNN Model** (our proposed architecture)
2. **VGG16** - Deep CNN known for simplicity and effectiveness in medical imaging
3. **EfficientNetB0** - Balanced architecture with compound scaling for efficiency
4. **MobileNetV2** - Lightweight model optimized for deployment on resource-limited devices

All pre-trained models were fine-tuned using transfer learning with custom regression heads adapted to our bone age prediction task. The comparison evaluated:
- Prediction accuracy (MAE)
- Generalization capability across datasets
- Training efficiency and convergence
- Computational requirements

**Our custom model was selected as the final solution** due to its superior performance on both RSNA and Bainem datasets, better generalization to local population characteristics, and efficient architecture designed specifically for medical imaging tasks.

## Performance Metrics

| Model | MAE on RSNA (months) | MAE on Bainem (months) |
|-------|---------------------|------------------------|
| **Custom Model** | **10.38** | **12.86** |
| EfficientNetB0 | 15.88 | 14.87 |
| VGG16 | 14.72 | 44.15 |
| MobileNetV2 | 16.46 | 29.75 |

- **R² Score**: 0.75 on Bainem dataset
- **Validation**: Meets clinical standards (6-12 months MAE target)

## Data Preprocessing

The preprocessing pipeline includes:

- Resizing with padding to 224×224 pixels
- Z-score normalization
- Data augmentation (horizontal flipping, brightness/contrast adjustment, rotation ±10°)
- Gender encoding (binary: 0=Female, 1=Male)
- Age normalization to [0, 1] interval

## GUI Application

A user-friendly interface built with PyQt5 allows clinicians to:

- Load hand radiographs (.png format)
- Select patient gender
- Obtain instant bone age predictions
- View results in a clinical-friendly format

The application is packaged as a standalone Windows executable (.exe) for easy deployment.

## Datasets

### RSNA Dataset
- **Source**: Radiological Society of North America
- **Size**: 12,611 left-hand radiographs
- **Age Range**: 0-228 months

### Bainem Hospital Dataset
- **Source**: Beni Messous Hospital, Algiers, Algeria
- **Size**: 173 radiographs
- **Purpose**: Local population adaptation and validation


## Contributors

- **BENAICHA CHAHINEZ** 
- **BENTAIFOUR YAMINA**
- **TARAFAT MELISSA**
- **BENABDALLAH HADHER**

## Acknowledgments

- Radiological Society of North America (RSNA) for the public dataset
- Beni Messous (Bainem) Hospital, Algiers, for local data collaboration
- Academic supervisors and clinical advisors


**Note**: This project is intended for research and educational purposes. Clinical deployment requires proper validation, regulatory approval, and medical oversight.
