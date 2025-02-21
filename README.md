# Melanoma Detection Assignment

## Table of Contents
- Problem Overview
- Objectives
- Methodology
- Technologies & Libraries Used
- Findings & Insights
- Acknowledgements
- Glossary
- Author

## Problem Overview
The goal of this project is to develop a custom Convolutional Neural Network (CNN) model for the detection of melanoma, a potentially fatal form of skin cancer. Given that melanoma accounts for a significant proportion of skin cancer-related fatalities, early detection can play a crucial role in improving patient outcomes. The model aims to analyze skin lesion images and assist dermatologists in making timely and accurate diagnoses.

The dataset used in this project consists of images classified into malignant and benign oncological conditions. These images have been categorized based on established dermatological classifications. The dataset includes multiple skin conditions, such as:
- Actinic Keratosis
- Basal Cell Carcinoma
- Dermatofibroma
- Melanoma
- Nevus
- Pigmented Benign Keratosis
- Seborrheic Keratosis
- Squamous Cell Carcinoma
- Vascular Lesions

### Important Considerations:
- The use of pre-trained models via transfer learning is not permitted; all models must be built from scratch.
- Some advanced topics introduced in the project are designed to facilitate deeper learning.
- Given the dataset size and training complexity, leveraging a GPU runtime (such as in Google Colab) is recommended.

## Objectives
The primary aim of this project is to design and implement a deep learning model capable of classifying images of skin lesions with high accuracy. Specific objectives include:
- Developing a CNN architecture tailored for melanoma detection.
- Reducing manual diagnostic workload through automated image classification.
- Experimenting with model optimization techniques to improve performance.
- Addressing challenges such as class imbalance and overfitting.

## Methodology
### Steps Followed:
1. **Importing Required Libraries**
2. **Loading and Understanding the Dataset**
3. **Preprocessing Data: Train-Validation Split & Test Set**
4. **Constructing the Baseline Model (Model 1)**
5. **Enhancing Performance Using Data Augmentation**
6. **Addressing Class Imbalance and Building an Optimized Model**
7. **Fine-tuning the Model and Hyperparameter Optimization**
8. **Evaluating Model Performance Through Predictions**
9. **Drawing Conclusions Based on Comparative Analysis**

## Technologies & Libraries Used
- **numpy** : 1.26.4
- **pandas** : 2.2.2
- **matplotlib** : 3.7.1
- **tensorflow** : 2.17.0
- **keras** : 3.4.1
- **augmentor** : 0.2.12

## Findings & Insights
Following a comprehensive evaluation of multiple CNN models, key insights include:
- A deep CNN model with ten convolutional layers, dropout layers, batch normalization, and controlled learning rates demonstrated the best performance, achieving **validation accuracy of 0.73** and **test accuracy of 0.50**.
- Intermediate models with moderate complexity achieved validation accuracies ranging between **0.65 and 0.70**, highlighting the importance of a well-balanced architecture.
- Simpler baseline models exhibited overfitting, as indicated by high training accuracy but significantly lower test accuracy.
- Techniques such as **data augmentation, class balancing, and dropout regularization** significantly improved generalization capability.
- Further optimization opportunities exist to enhance performance, including **fine-tuning learning rates and adjusting the number of epochs**.

### Frequently Asked Questions:
- **Which class had the least number of samples?**
  - Seborrheic Keratosis.
- **Which class was the most dominant?**
  - Pigmented Benign Keratosis.

## Glossary
- **Data Augmentation** – Techniques to artificially expand a dataset by applying transformations.
- **Class Imbalance** – A situation where certain categories have significantly fewer examples than others.
- **Train-Validation Split** – Dividing data into separate sets for training and validation.
- **Test Set** – A separate dataset used to evaluate model performance.
- **Convolutional Neural Network (CNN)** – A deep learning model specifically designed for image processing.
- **Dropout** – A regularization technique that prevents overfitting by randomly deactivating neurons during training.
- **Learning Rate (LR)** – A hyperparameter that controls how much a model updates its weights in response to errors.
- **Overfitting** – When a model performs well on training data but poorly on unseen data.
- **Early Stopping** – A technique that halts training when performance no longer improves.
- **Cross-Entropy Loss** – A loss function commonly used for classification problems.
- **Accuracy** – A metric that measures how often a model makes correct predictions.
- **Batch Normalization** – A technique that stabilizes learning and speeds up convergence.
- **Max Pooling** – A downsampling technique used in CNNs to reduce spatial dimensions.
- **Softmax** – A function that converts logits into probabilities in multi-class classification.
- **Learning Rate Scheduler (ReduceLROnPlateau)** – A method that adjusts the learning rate dynamically during training.

