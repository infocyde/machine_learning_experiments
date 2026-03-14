import utils as ut
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import tensorflow as tf


def main():
    
    ut.clear_console()
    
    iris = load_iris()
    x_train, x_test, y_train, y_test = train_test_split(
        iris.data, iris.target, test_size=0.2, random_state=42)
    
    # build a simple neural network model
    # creates four layers, first three are hidden layers with relu activation, last layer is output layer with softmax for multi-class classification
    # relu helps the model learn complex patterns, softmax gives probabilities for each class, relu stands for rectified linear unit
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)), # input layer, 4 features in the dataset, subsequent layers infer from this
        tf.keras.layers.Dense(20, activation='relu'),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

 
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    #train the model
    model.fit(x_train, y_train, epochs=100, verbose=0) #epochs is how many times to go through the training data, verbose=0 means no output during training
    predictions = model.predict(x_test)
    predicted_classes = predictions.argmax(axis=1)
    
    score = metrics.accuracy_score(y_test, predicted_classes)
    print('Accuracy: {0:f}'.format(score))



if __name__ == '__main__':
    main()