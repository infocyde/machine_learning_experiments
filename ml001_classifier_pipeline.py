from sklearn import datasets
import utils as ut

ut.clear_console()
print("starting classifier example")


iris = datasets.load_iris()  # wildlife dataset built into sklearn

# f(x) = y
x = iris.data  # features
y = iris.target  # labels

#now let's see if they are actually matching (test half of them)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.5, random_state=4)  # think random_state = lock the shuggle order, not set, it is random every time test_size = 0.5 means half for training, half for testing

from sklearn import tree    
clf = tree.DecisionTreeClassifier()

clf = clf.fit(X_train, y_train)  #note the fit function is where the training happens, x is what to train on, features, y is the labels (what it should predict) so you are forcing the classifier to learn the relationship between x and y

predictions = clf.predict(X_test)  #note the predict function is where the testing happens, it returns an array of predictions

ut.clear_console()
print("Tree Predictions:", predictions)


from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, predictions)  #compare what it predicted to what is actually true
print("\nTree Accuracy:", accuracy)


# now lets do it with k nearest neighbors, more accurate but slower
from sklearn.neighbors import KNeighborsClassifier
clf2 = KNeighborsClassifier(n_neighbors=3)  # k is how many neighbors to consider when making a decision
clf2 = clf2.fit(X_train, y_train)
predictions2 = clf2.predict(X_test)
accuracy2 = accuracy_score(y_test, predictions2)
print("\nKNN Predictions:", predictions2)
print("\nKNN Accuracy:", accuracy2)
