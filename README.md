# MAGIC Gamma vs Hadronic Classifier

This project classifies high-energy gamma particles and hadronic showers using machine learning algorithms. The data is simulated using Monte Carlo (MC) methods to represent the patterns captured by ground-based atmospheric Cherenkov telescopes such as the MAGIC telescope.

## Project Overview

Cherenkov telescopes detect high-energy gamma rays by observing the Cherenkov radiation emitted by charged particles produced in electromagnetic showers initiated by the gamma rays. These showers create distinct image patterns in the telescope's photomultiplier tubes (PMTs). The challenge is to distinguish between images caused by gamma-ray showers (signal) and those caused by hadronic showers (background noise).

This project implements multiple machine learning classification algorithms to achieve this goal, including:
- k-Nearest Neighbors (kNN)
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)
- Neural Networks

The project uses various features extracted from the shower images, known as Hillas parameters, to classify the events.

## Dataset

The dataset used in this project is from the UCI MAGIC Gamma Telescope dataset, consisting of 10 features and a target class (`gamma` or `hadron`). It is available for download [here](https://archive.ics.uci.edu/ml/datasets/magic+gamma+telescope).

**Features:**
- `fLength`: Major axis of the ellipse (size)
- `fWidth`: Minor axis of the ellipse (size)
- `fSize`: Integrated light content
- `fConc`: Concentration (ratio of brightest pixel to total content)
- `fConc1`: Similar to `fConc` but for 2 brightest pixels
- `fAsym`: Asymmetry of the light distribution
- `fM3Long`: 3rd root of third moment along major axis
- `fM3Trans`: 3rd root of third moment along minor axis
- `fAlpha`: Angle of the major axis with the camera's center
- `fDist`: Distance between the major axis and the camera's center

The target feature is binary (`gamma` or `hadron`), which is mapped to 1 for gamma and 0 for hadron.

## Model Workflow

1. **Data Preprocessing**: The dataset is preprocessed by standardizing the features using `StandardScaler`. Oversampling using `RandomOverSampler` is performed to handle the class imbalance.
   
2. **Training and Testing Split**: The dataset is split into training (60%), validation (20%), and test (20%) sets.

3. **Model Training**: Several classifiers are trained on the data, including kNN, Naive Bayes, Logistic Regression, SVM, and Neural Networks. The neural network model is tuned using different hyperparameters (number of neurons, dropout probability, learning rate, and batch size) to find the best-performing model.

4. **Evaluation**: Each model's performance is evaluated using classification metrics like precision, recall, F1-score, and accuracy. The results are compared to choose the best-performing model.

## Results

The Support Vector Machine and Neural Network models performed best, achieving an accuracy of approximately 87% on the test set. Precision, recall, and F1-score were used to assess the performance further, especially in cases of imbalanced data.

## Installation and Usage

1. Clone the repository.
2. Install the required packages using `requirements.txt`.
3. Load the dataset and preprocess it.
4. Train the machine learning models and evaluate their performance.
5. Fine-tune hyperparameters for the neural network model to improve performance.

## Acknowledgments

The dataset used in this project is sourced from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php).  
This project was developed following the live tutorial available [here](https://www.youtube.com/watch?v=i_LwzRVP7bg&t=9626s).  
A special thanks to Kylie Ying for providing in-depth knowledge of machine learning algorithms, including both the underlying math and hands-on coding.





