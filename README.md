# Genre Classification Project
## Abstract
This project focuses on classifying music genres using machine-learning techniques. 

## Implementation
The primary implementation involves data processing and feature extraction, specifically using Mel-Frequency Cepstral Coefficients (MFCCs). These features are fed into two distinct models: a Multi-Layer Perceptron (MLP) and a Convolutional Neural Network (CNN).

#### MLP Model
The MLP model is a sequential neural network with multiple dense layers. It comprises an input layer, three hidden layers with 1024, 512, and 256 units respectively, a dropout layer to prevent overfitting, and an output layer with 10 units representing the genres. The activation function used is ReLU for the hidden layers and softmax for the output layer. The model is compiled with the Adam optimizer, a learning rate of 0.0001, and sparse categorical crossentropy as the loss function.

#### CNN Model 
The CNN model architecture comprises three convolutional layers with 32 filters each, utilizing ReLU activation, max pooling, and batch normalization for feature extraction. The final layers include a flattening operation, a dense layer with 64 units and ReLU activation, and an output layer with 10 units for genre classification using softmax activation. Dropout is applied to prevent overfitting. This architecture is designed to capture spatial patterns in audio data, providing an effective framework for genre classification.

#### Training
Both models are trained on the dataset, and the training process includes validation on a separate test set.
