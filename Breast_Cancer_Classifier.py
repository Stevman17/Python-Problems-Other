import codecademylib3_seaborn
#We want to import the function load_breast_cancer from sklearn.datasets.
from sklearn.datasets import load_breast_cancer
#Once we've imported the dataset, let's load the data into a variable called breast_cancer_data
breast_cancer_data = load_breast_cancer()
print(breast_cancer_data.data[0])
print(breast_cancer_data.feature_names)
print(breast_cancer_data.target)
print(breast_cancer_data.target_names)
#We have our data, but now it needs to be split into training and validation sets.
from sklearn.model_selection import train_test_split
#train_test_split returns four values in the following order:
#The training set
#The validation set
#The training labels
#The validation labels
#Store those values in variables named training_data, validation_data, training_labels, and validation_labels.
training_data, validation_data, training_labels,  validation_labels = train_test_split(breast_cancer_data.data, breast_cancer_data.target, test_size=0.2, random_state=100)
from sklearn.neighbors import KNeighborsClassifier 
#Create a KNeighborsClassifier where n_neighbors = 3. Name the classifier classifier.
classifier = KNeighborsClassifier(n_neighbors = 3)
#Train your classifier using the fit function. This function takes two parameters: the training set and the training labels.
classifier.fit(training_data, training_labels)
#Now that the classifier has been trained, let's find how accurate it is on the validation set. Call the classifier's score function. score takes two parameters: the validation set and the validation labels. 
print(classifier.score(validation_data, validation_labels))
#The classifier does pretty well when k = 3. But maybe there's a better k! Put the previous 3 lines of code inside a for loop. The loop should have a variable named k that starts at 1 and increases to 100. Rather than n_neighbors always being 3, it should be this new variable k.
accuracies = []
for k in range(1, 101):
  classifier = KNeighborsClassifier(n_neighbors = k)
  classifier.fit(training_data, training_labels)
  accuracies.append(classifier.score(validation_data,validation_labels))
#You should now see 100 different validation accuracies print out. Which k seems the best?
import matplotlib.pyplot as plt
#The y-axis of our graph should be the validation accuracy. Instead of printing the validation accuracies, we want to add them to a list. Outside of the for loop, create an empty list named accuracies. Inside the for loop, instead of printing each accuracy, append it to accuracies.
k_list = range(1,101)
#We can now plot our data! Call plt.plot(). The first parameter should be k_list and the second parameter should be accuracies.

plt.plot(k_list, accuracies)
plt.show()
