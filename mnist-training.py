#%%
import numpy as numpy
import keras as keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.datasets import mnist

#%%
# load data from mnist and explore it
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# make sure our data is from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print("Training images shape: " + str(train_images.shape))
print("Training labels shape: " + str(train_labels.shape))
print("Test images shape: " + str(test_images.shape))
print("Test labels shape: " + str(test_labels.shape))

#%%
# do some data preprocessing and see what an training image looks like
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i+1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(train_labels[i])

#%%
