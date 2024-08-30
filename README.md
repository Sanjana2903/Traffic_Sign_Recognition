# Traffic_Sign_Recognition
This project aims to classify German traffic signs using a Convolutional Neural Network (CNN) built with TensorFlow and Keras. The model is trained and tested on the German Traffic Sign Recognition Benchmark (GTSRB) dataset and is capable of recognizing 43 different traffic sign classes with high accuracy.

This project uses deep learning techniques to classify traffic signs from the GTSRB dataset. The CNN model processes image data to accurately predict the type of traffic sign from 43 classes. The model is designed using TensorFlow and Keras, and it achieves significant accuracy on both training and unseen test data.

**Features**

End-to-end CNN model for traffic sign classification.
Data preprocessing: resizing images and converting labels to one-hot encoding.
Trained and evaluated with visualization of training performance (accuracy and loss curves).
Capable of predicting traffic signs in real-world test images.

**Installation**

1. Clone this repository
2. Install the required dependencies
pip install numpy pandas tensorflow Pillow matplotlib scikit-learn opencv-python
3. Download the dataset 

**Model Architecture**

The CNN model includes the following layers:

Conv2D: Convolutional layers for feature extraction with ReLU activation.
MaxPool2D: Pooling layers to reduce spatial dimensions.
Dropout: Regularization to prevent overfitting.
Flatten: Flattening the 2D features to 1D.
Dense: Fully connected layers with ReLU and softmax activations.
Optimizer: Adam
Loss Function: Categorical Crossentropy
Metrics: Accuracy
Epochs: 15

**Training**

1. Preprocess the image data by resizing all images to 30x30 pixels and splitting them into training and testing sets.
2. The model is trained using 32-batch sizes for 15 epochs, leveraging GPU acceleration if available.
3. The training process includes monitoring accuracy and loss for both the training and validation datasets.

**Evaluation and Results**

The model is evaluated using the test set. The performance metrics include accuracy and loss on unseen data. Visualizations of the accuracy and loss curves over the training epochs are provided.
