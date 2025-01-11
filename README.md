# Damage Prediction and Traction-Separation Analysis

## Overview
This project involves:
- Generating a dataset to simulate damage and traction-separation behavior using mathematical models.
- Visualizing traction-separation laws and damage-separation relationships.
- Implementing and training Artificial Neural Networks (ANNs) for damage prediction.
- Evaluating model performance and visualizing results.

## Requirements
### Libraries
The following Python libraries are required:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `tensorflow`

### Installation
Install the required libraries using:
`
pip install numpy pandas scikit-learn matplotlib seaborn tensorflow
`
### Dataset Generation

Mathematical Models:
damage_non_linear: Models non-linear damage behavior based on parameters.
traction_linear: Computes traction from separation and damage values.
Parameters:
sigma_max_values: Ranges for maximum stress.
delta_u_values: Ranges for ultimate separation.
alpha_values: Exponential factors for non-linear damage modeling.
Output:
The generated dataset is saved as data.csv.
Train-test splits are saved as:
X_train.csv
X_test.csv
y_train.csv
y_test.csv
Visualizations

### Traction-Separation Law:
Visualizes the relationship between separation (δ) and traction (T).
Output: Traction-Separation Law (TSL).png
Damage vs Separation:
Visualizes the relationship between separation (δ) and damage (D).
Output: Damage vs Separation.png
Model Training

1. Scikit-learn ANN
Implementation:
Built using MLPRegressor with 3 hidden layers.
Optimized using the Adam optimizer.
Outputs:
Loss curve and actual vs predicted damage values.
Visualized and saved as ANN_p2.png.
Metrics:
Train Loss, Test Loss, and Mean Absolute Error (MAE).
2. TensorFlow ANN
Implementation:
Sequential model with 3 hidden layers using ReLU activation.
Trained using Adam optimizer.
Outputs:
Loss and MAE curves (training and validation).
Visualized and saved as ANN.png.
Metrics:
Test Loss and Test MAE.
3. Custom ANN
Implementation:
A 4-layer ANN implemented from scratch with NumPy.
Uses ReLU activation and Mean Squared Error (MSE) loss.
Outputs:
Training loss curve and actual vs predicted damage values.
Visualized and saved as ANN_p.png.
Metrics:
Test Loss and Test MAE.

