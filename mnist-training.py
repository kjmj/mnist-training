#%%
import numpy as numpy
import keras as keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.datasets import mnist

#%%
# load data from mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# make sure our data is from 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

print('Training images shape:', str(train_images.shape))
print('Training labels shape:', str(train_labels.shape))
print('Test images shape:', str(test_images.shape))
print('Test labels shape:', str(test_labels.shape))

#%%
# do some data preprocessing to see what our images and labels look like
plt.figure(figsize=(10, 10))
for i in range(25):
  plt.subplot(5, 5, i + 1)
  plt.xticks([])
  plt.yticks([])
  plt.grid(False)
  plt.imshow(train_images[i], cmap=plt.cm.binary)
  plt.xlabel(train_labels[i])

#%%
# build, compile and train the model
model = keras.Sequential([
  keras.layers.Flatten(input_shape=(28, 28)), # transform our 2d array to a flat array
  keras.layers.Dense(128, activation='relu'),
  keras.layers.Dense(10, activation='softmax') # will return our probability scores
])

model.compile(loss='sparse_categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)

model.fit(train_images, train_labels, epochs=5)

#%%
# take a look at the models accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
print('Test loss:', test_loss)

#%%
# make predictions
predictions = model.predict(test_images)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img, cmap=plt.cm.binary)
  
  predicted_label = numpy.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'green'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(predicted_label,
                                100*numpy.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = numpy.argmax(predictions_array)
  
  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('green')

# Plot the first X test images, their predicted label, and the true label
# Color correct predictions in blue, incorrect predictions in red
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions, test_labels)
plt.show()

#%%
