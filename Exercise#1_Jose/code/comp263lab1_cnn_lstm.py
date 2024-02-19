
# -*- coding: utf-8 -*-
"""

COMP263LAB1_CNN_LSTM.py

jose muniz
301316969


"""

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical,plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Import and load the 'fashion_mnist' dataset from TensorFlow
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images.shape,test_images.shape

# Store the datasets into dictionaries
train_JOSE = {'images': train_images, 'labels': train_labels}
test_JOSE = {'images': test_images, 'labels': test_labels}

# b: Initial Exploration
print("Size of training dataset:", len(train_JOSE['images']))
print("Size of testing dataset:", len(test_JOSE['images']))
print("Image resolution (dimension) of input images:", train_JOSE['images'].shape[1:3])
print("Largest pixel value in the dataset:", np.amax(train_JOSE['images']))

#c: Data Pre-preprocessing
# Normalize the pixel values
train_JOSE['images'] = train_JOSE['images'] / 255.0
test_JOSE['images'] = test_JOSE['images'] / 255.0

# One-hot encode the labels
train_JOSE['labels'] = to_categorical(train_JOSE['labels'])
test_JOSE['labels'] = to_categorical(test_JOSE['labels'])

# Display the shape of the labels
print("Shape of train_JOSE['labels']:", train_JOSE['labels'].shape)
print("Shape of test_JOSE['labels']:", test_JOSE['labels'].shape)

test_JOSE['images'], test_JOSE['labels']

#d: Visualization
# Function to display an image with its label
def plot_image(image, label):
    plt.imshow(image, cmap=plt.cm.binary)
    plt.title(np.argmax(label))
    plt.xticks([])
    plt.yticks([])

# Plot the first 12 data samples
plt.figure(figsize=(8, 8))
for i in range(12):
    plt.subplot(4, 3, i+1)
    plot_image(train_JOSE['images'][i], train_JOSE['labels'][i])
plt.show()

#e: Training Data Preparation
# Split the training dataset
x_train_JOSE, x_val_JOSE, y_train_JOSE, y_val_JOSE = train_test_split(
    train_JOSE['images'], train_JOSE['labels'], test_size=0.2, random_state=69)

#f: Build, Train, and Validate CNN Model
# Building the CNN model
cnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Conv2D(32, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model
cnn_model.summary()

#plot_model(cnn_model, to_file='model.png', show_shapes=True, show_layer_names=True)

# Train and validate the model
train_history = cnn_model.fit(x_train_JOSE, y_train_JOSE, batch_size=256,
                          validation_data=(x_val_JOSE, y_val_JOSE), epochs=10)

# g: Test and analyze the model ---
# Plotting Training Vs Validation Accuracy
plt.figure(figsize=(8, 4))
plt.plot(train_history.history['accuracy'], label='Training Accuracy')
plt.plot(train_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model with test data
test_loss, test_accuracy = cnn_model.evaluate(test_JOSE['images'], test_JOSE['labels'])
print("Test accuracy:", test_accuracy)

# Evaluate using test data
#Reshape test images for prediction
test_images_reshaped = test_JOSE['images'].reshape(-1, 28, 28, 1)

# Save the CNN model
cnn_model.save('cnn_model.h5')

# Create predictions on the test dataset
predictions = cnn_model.predict(test_images_reshaped)

# Function to plot prediction distribution
def plot_prediction_distribution(image, true_label, prediction):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'True Label: {np.argmax(true_label)}')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction)
    plt.xticks(range(10))
    plt.ylim([0, 1])
    true_label_index = np.argmax(true_label)
    predicted_label_index = np.argmax(prediction)
    plt.bar(true_label_index, prediction[true_label_index], color='green')
    plt.bar(predicted_label_index, prediction[predicted_label_index], color='blue')
    plt.title(f'Predicted Label: {predicted_label_index}')
    plt.show()


for i in range(4):
    plot_prediction_distribution(test_JOSE['images'][i], test_JOSE['labels'][i], predictions[i])

# Convert predictions to label indices
predicted_labels = [np.argmax(p) for p in predictions]
true_labels = [np.argmax(l) for l in test_JOSE['labels']]

# Print classification report
print(classification_report(true_labels, predicted_labels))

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

"""The model present a good prediction for the mayorities label. (diagonal line)

---
The level number 6 show the mayor misclassification. It is often confused with Class 0 , Class 2 , and Class 4 .

---

Class 1 and Class 5  and 7 are the best predicted with very few misclassifications, indicating the model can easily distinguish these items from others.

---



"""

# H : Build,Train,Validate,Test and Analyze RNN Model

# Reshape images for RNN input (considering each row as a timestep)
x_train_rnn = x_train_JOSE.reshape(x_train_JOSE.shape[0], 28, 28)
x_val_rnn = x_val_JOSE.reshape(x_val_JOSE.shape[0], 28, 28)
test_images_rnn = test_JOSE['images'].reshape(test_JOSE['images'].shape[0], 28, 28)

# Building the RNN model
rnn_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.LSTM(128),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the RNN model
rnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the RNN model
rnn_model.summary()

#plot_model(rnn_model, to_file='rnn_model.png', show_shapes=True, show_layer_names=True)

# Train and validate the RNN model
rnn_history = rnn_model.fit(x_train_rnn, y_train_JOSE, batch_size=256,
                            validation_data=(x_val_rnn, y_val_JOSE), epochs=10)

# Evaluate the RNN model with test data
test_loss_rnn, test_accuracy_rnn = rnn_model.evaluate(test_images_rnn, test_JOSE['labels'])
print("RNN Test accuracy:", test_accuracy_rnn)

# Save the RNN model
rnn_model.save('rnn_model.h5')

# Create predictions on the test dataset using the RNN model
rnn_predictions = rnn_model.predict(test_images_rnn)

# Plotting Training Vs Validation Accuracy for RNN Model
plt.figure(figsize=(8, 4))
plt.plot(rnn_history.history['accuracy'], label='Training Accuracy')
plt.plot(rnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('RNN Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Create predictions on the test dataset
predictions_rnn = rnn_model.predict(test_images_rnn)

# Function to plot prediction distribution
def plot_prediction_distribution(image, true_label, prediction):
    plt.figure(figsize=(12, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.title(f'True Label: {np.argmax(true_label)}')

    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction)
    plt.xticks(range(10))
    plt.ylim([0, 1])
    true_label_index = np.argmax(true_label)
    predicted_label_index = np.argmax(prediction)
    plt.bar(true_label_index, prediction[true_label_index], color='green')
    plt.bar(predicted_label_index, prediction[predicted_label_index], color='blue')
    plt.title(f'Predicted Label: {predicted_label_index}')
    plt.show()


for i in range(4):
    plot_prediction_distribution(test_JOSE['images'][i], test_JOSE['labels'][i], predictions_rnn[i])

# Convert predictions to label indices
predicted_labels = [np.argmax(p) for p in predictions_rnn]
true_labels = [np.argmax(l) for l in test_JOSE['labels']]

# Print classification report
print(classification_report(true_labels, predicted_labels))

# Plot the confusion matrix
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()