import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Updated import
import os
from zipfile import ZipFile
import matplotlib.pyplot as plt
import seaborn as sns

# Check TensorFlow version
print(f"TensorFlow version: {tf.__version__}")

# Data Preprocessing
# Replace with the actual paths on your local system
base_dir = 'dataset'

# ImageDataGenerator for training and test sets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

training_set = train_datagen.flow_from_directory(
    os.path.join(base_dir, 'training_set'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

test_datagen = ImageDataGenerator(rescale=1./255)

test_set = test_datagen.flow_from_directory(
    os.path.join(base_dir, 'test_set'),
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)

# Building the CNN
cnn = tf.keras.models.Sequential()

cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=[64, 64, 3]))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Conv2D(filters=32, kernel_size=3, activation='relu'))
cnn.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
cnn.add(tf.keras.layers.Flatten())
cnn.add(tf.keras.layers.Dense(units=128, activation='relu'))
cnn.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the CNN
cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training the CNN and storing the history for visualization
history = cnn.fit(x=training_set, validation_data=test_set, epochs=25)

# Saving the trained model in .h5 format
model_save_path = 'saved_model/my_cnn_model.h5'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
cnn.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# Visualize training and validation accuracy and loss
sns.set(style='whitegrid')

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training and Validation Metrics')

# Accuracy plot
axs[0].plot(history.history['accuracy'], label='Training Accuracy')
axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
axs[0].set_title('Model Accuracy')
axs[0].set_xlabel('Epoch')
axs[0].set_ylabel('Accuracy')
axs[0].legend()

# Loss plot
axs[1].plot(history.history['loss'], label='Training Loss')
axs[1].plot(history.history['val_loss'], label='Validation Loss')
axs[1].set_title('Model Loss')
axs[1].set_xlabel('Epoch')
axs[1].set_ylabel('Loss')
axs[1].legend()

plt.show()
