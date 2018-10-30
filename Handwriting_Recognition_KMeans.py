import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans

digits = datasets.load_digits()
print(digits.data)
print(digits.target)
# Figure size (width, height)
fig = plt.figure(figsize=(6, 6))
# Adjust the subplots 
fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
# For each of the 64 images
for i in range(64):
    # Initialize the subplots: add a subplot in the grid of 8 by 8, at the i+1-th position
    ax = fig.add_subplot(8, 8, i+1, xticks=[], yticks=[])
    # Display an image at the i-th position
    ax.imshow(digits.images[i], cmap=plt.cm.binary, interpolation='nearest')
    # Label the image with the target value
    ax.text(0, 7, str(digits.target[i]))

#Q: What should be the k, the number of clusters, here?
#A: 10, for 10 numbers

#Use the KMeans() method to build a model that finds k clusters.
model = KMeans(n_clusters = 10, random_state=42)
#The random_state will ensure that every time you run your code, the model is built in the same way. This can be any number. We used random_state = 42.

#Use the .fit() method to fit the digits.data to the model.
model.fit(digits.data)
#Let's visualize all the centroids! Because data samples live in a 64-dimensional space, the centroids have values so they can be images!
#First, add a figure of size 8x3 using .figure().
fig = plt.figure(figsize=(8, 3))
#Then, add a title using .suptitle().
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')
#Write a for loop to displays each of the cluster_centers:
for i in range(10):
  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)
  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()
#draw 4 integers and convert them to a 2d arrary
#Back in script.py, create a new variable named new_samples and copy and paste the 2D array into it.
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,4.39,6.10,6.10,6.10,5.40,2.26,0.00,0.00,3.19,4.57,4.57,5.33,7.62,5.54,0.00,0.00,0.00,0.00,0.00,1.76,7.62,3.20,0.00,0.00,0.00,1.94,7.24,7.62,7.62,4.70,0.00,0.00,0.00,2.58,7.51,7.62,5.55,2.26,0.00,0.00,0.00,0.00,3.71,7.62,1.78,0.00,0.00,0.00,0.00,0.00,4.87,6.15,0.00,0.00,0.00],
[0.00,0.00,5.15,7.22,7.62,7.62,7.62,7.62,0.00,0.00,4.34,4.55,3.74,3.12,7.31,6.61,0.00,0.00,0.00,0.00,4.39,6.25,7.62,5.56,0.00,0.00,0.00,0.00,3.18,7.07,7.47,7.54,0.00,0.00,0.00,0.00,1.44,7.55,4.05,0.45,0.00,0.00,0.00,0.43,7.17,7.10,0.70,0.00,0.00,0.00,0.00,0.15,3.59,1.12,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,2.19,5.34,3.97,1.52,0.08,0.00,0.00,0.00,2.22,5.72,7.55,7.62,6.71,0.58,0.00,0.00,0.00,0.00,0.69,3.87,7.62,3.81,0.00,0.00,0.00,3.18,6.10,6.33,7.55,2.27,0.00,0.00,0.00,2.29,5.85,7.62,7.62,7.62,0.00,0.00,0.00,0.00,3.71,7.63,2.29,1.52,0.00,0.00,0.00,0.00,6.17,6.03,0.00,0.00,0.00,0.00,0.00,1.70,7.62,3.50,0.00,0.00],
[0.00,0.08,1.35,0.30,0.00,0.00,0.00,0.00,0.00,2.13,7.62,7.62,6.71,5.56,4.34,1.12,0.00,0.22,2.72,3.81,5.03,6.45,7.62,3.43,0.00,0.00,0.00,0.00,0.69,3.33,7.62,1.90,0.00,0.00,1.75,7.60,7.62,7.62,7.62,6.15,0.00,0.00,0.46,3.02,4.65,7.62,7.62,3.97,0.00,0.00,0.00,0.00,4.19,7.62,7.62,0.51,0.00,0.00,0.00,0.00,4.80,6.38,4.57,0.00]

])
new_labels = model.predict(new_samples)
print(new_labels)
#By looking at the cluster centers, let's map out each of the labels with the digits we think it represents:
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

