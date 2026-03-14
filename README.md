# machine_learning_experiments

A collection of Python scripts exploring machine learning classification techniques, progressing from basic scikit-learn classifiers to neural networks with TensorFlow.

## Scripts

### ml001_classifier.py

Introduces the simplest possible classifier — a Decision Tree using scikit-learn. Trains on a tiny hardcoded dataset of fruit features (weight and texture) to classify apples vs oranges. Demonstrates the basic `fit` / `predict` workflow.

### ml001_classifier_adv.py

Expands on the basic classifier by using the Iris dataset and visualizing the trained Decision Tree with Graphviz. Splits data into training and test sets manually, trains a `DecisionTreeClassifier`, evaluates predictions, and renders the full decision tree as a PDF.

### ml001_classifier_pipeline.py

Compares two scikit-learn classifiers side-by-side on the Iris dataset: `DecisionTreeClassifier` and `KNeighborsClassifier`. Uses `train_test_split` for a proper 50/50 train-test split and reports accuracy scores for both approaches.

### ml001_classifier_from_scratch.py

Implements a K-Nearest Neighbors (KNN) classifier from scratch (`ScrappyKNN`) using Euclidean distance, without relying on scikit-learn's built-in KNN. Also includes a random-baseline classifier (`ScrappyKNN2`) for comparison. Tests the custom classifier on the Iris dataset to demonstrate that the hand-built version achieves comparable accuracy.

### ml001_tensorflow_example.py

Builds a simple dense neural network with TensorFlow/Keras to classify the Iris dataset. Uses a Sequential model with three hidden layers (ReLU activation) and a softmax output layer for 3-class classification. Trains for 100 epochs and reports accuracy.

### ml001_cnn_text_classifier.py

Builds a Convolutional Neural Network (CNN) with TensorFlow/Keras to classify handwritten digits from the MNIST dataset. Uses three convolutional blocks with max-pooling, a dropout layer for regularization, and a softmax output for 10-class digit classification. Trains for 5 epochs and evaluates test accuracy.

### utils.py

Shared utility module providing a `clear_console()` helper function used across the other scripts.
