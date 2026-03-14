from sklearn import datasets
#from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import random, numpy as np

from scipy.spatial import distance 

import utils as ut


def euc(a, b):  #using euclidean distance to measure distance between points
    return distance.euclidean(a, b)  #might need something more complex if doing more axis than just too.

# actuall implementation of kNN algorithm from scratch
class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self # necessary else error


    def predict(self, X_test):
        predictions = []
        for row in X_test:
            # find the closest training example
            label = self.closest(row) # actual nearest neighbor logic, defaulting to just 1 nearest neighbor
            predictions.append(label)
        #return predictions 
        return np.array(predictions) # cleaner for printing etc else just get the numpy array representation for each list item
    

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]


# basic inferface for classifiers, fit and predict methods, not used now but good for future reference
class ScrappyKNN2():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        return self # necessary else error


    def predict(self, X_test):
        predictions = []
        for row in X_test:
            # find the closest training example
            label = random.choice(self.y_train)  # random choice as a placeholder for actual nearest neighbor logic
            predictions.append(label)
        #return predictions 
        return np.array(predictions) # cleaner for printing etc else just get the numpy array representation for each list item
    
    



ut.clear_console()
print("starting classifier example")





iris = datasets.load_iris()  # wildlife dataset built into sklearn

# f(x) = y
x = iris.data  # features
y = iris.target  # labels

#now let's see if they are actually matching (test half of them)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)  # think random_state = lock the shuggle order, not set, it is random every time test_size = 0.5 means half for training, half for testing


# now lets do it with k nearest neighbors, more accurate but slower

clf = ScrappyKNN()  # k is how many neighbors to consider when making a decision
clf = clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("\nKNN Predictions:", predictions)
print("\nKNN Accuracy:", accuracy)
