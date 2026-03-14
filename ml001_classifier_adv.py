#pip install matplotlib scikit-learn  graphviz

from sklearn import datasets
from sklearn import tree
import matplotlib.pyplot as plt

import graphviz # to use this you musting install graphviz https://graphviz.org/download/ and then pip install graphviz, make sure on install it adds path 


import numpy as np  
import os



iris = datasets.load_iris() #wildlife dataset built into sklearn


# from series https://www.youtube.com/watch?v=tNa99PG8hR8&list=PLOU2XLYxmsIIuiBfYad6rFYQU_jL2ryal&index=2

# clear previous run
os.system('cls' if os.name == 'nt' else 'clear')



# # features and labels
print(iris.feature_names)  # equivalent to attributes of the class or all the rows that id a object
print(iris.target_names)   # class or object name equivalent (what the above identifies)

print("\nExamples:")

for i in range(len(iris.data)):
    print(f"Example {i+1}: Feature={iris.data[i]}, Label={iris.target[i]}")  # remember target is numeric, need to map that to meaningful names using target_names


# remove some items for testing, in this example 1 of each type of flower
test_indices = [0, 50, 100]
train_target = np.delete(iris.target, test_indices)
train_data = np.delete(iris.data, test_indices, axis=0)

# grab the three rows we removed earlier for testing
test_target = iris.target[test_indices]
test_data = iris.data[test_indices]

# train the classifier
clf = tree.DecisionTreeClassifier()
clf = clf.fit(train_data, train_target)  #note the fit function is where the training happens

print("\n training complete \n")

print("Test data: What is known to be true")
print(test_target)
print("What is predicted")
print(clf.predict(test_data))  #note the predict 

#simple visualization using matplotlib
# tree.plot_tree(clf)  # this will plot the decision tree used to make the decisions  
# plt.show()

# more advanced visualization using graphviz
dot_data = tree.export_graphviz(clf, out_file=None, 
                     feature_names=iris.feature_names,  
                     class_names=iris.target_names,  
                     filled=True, rounded=True,  
                     special_characters=True)  
graph = graphviz.Source(dot_data)  
graph 

graph.render("iris_decision_tree", view=True)  # will open the decision tree in default viewer

