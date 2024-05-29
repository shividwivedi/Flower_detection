import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import streamlit as st
import tensorflow as tf
import pathlib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.preprocessing.image import load_img, img_to_array


data_dir = pathlib.Path('./flowers')

# Print the data directory to verify the path
print(f"Data directory: {data_dir}")
#
# # (Optional) Check if the directory exists and print its contents
# if data_dir.exists() and data_dir.is_dir():
#     print("Directory exists. Here are the contents:")
#     # for item in data_dir.iterdir():
#     #     print(item)
# else:
#     print("Directory does not exist or is not a directory.")

img_height,img_width=180,180
batch_size=32
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
plt.show()

model = Sequential([
    # Add the Rescaling layer at the beginning
    Rescaling(1./255, input_shape=(img_height, img_width, 3)),

    # Convolutional Layer 1
    Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Convolutional Layer 2
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Convolutional Layer 3
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Convolutional Layer 4
    Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu'),
    MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),

    # Flattening Layer
    Flatten(),

    # Fully Connected Layer
    Dense(512, activation='relu'),

    # Output Layer
    Dense(22, activation="softmax")  # Adjust this if 22 is not the correct number of classes
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])

epochs=20
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

# Plotting training and validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()

# Predict classes for entire validation set
true_classes = []
pred_classes = []
for images, labels in val_ds:
    preds = model.predict(images)
    pred_classes.extend(np.argmax(preds, axis=1))
    true_classes.extend(labels.numpy())

# Generate the confusion matrix
cm = confusion_matrix(true_classes, pred_classes, labels=np.arange(len(class_names)))
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# Print classification report
print(classification_report(true_classes, pred_classes, target_names=class_names))
from tensorflow.keras.models import load_model
model.save('Model.h5')

# load model
savedModel=load_model('Model.h5')

#Input image
test_image = load_img('D:/Groq-llama3/pythonProject/flowers/daisy/5547758_eea9edfd54_n.jpg',target_size=(180,180))

#For show image
plt.imshow(test_image)
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image,axis=0)

# Result array
result = savedModel.predict(test_image)
print(result)

predicted_class_index = np.argmax(result, axis=1)[0]
print("Predicted class:", class_names[predicted_class_index])
#Mapping result array with the main name list
# i=0
# for i in range(len(result[0])):
#   if(result[0][i]==1):
#     print(list_[i])
#     break
