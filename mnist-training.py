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

print("Training images shape: " + str(train_images.shape))
print("Training labels shape: " + str(train_labels.shape))
print("Test images shape: " + str(test_images.shape))
print("Test labels shape: " + str(test_labels.shape))

#%%
# do some data preprocessing and see what an training image looks like
plt.figure()
plt.imshow(train_images[0], cmap='gray')
plt.colorbar()
plt.grid(False)
plt.show()
