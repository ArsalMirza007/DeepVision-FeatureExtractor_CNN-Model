# DeepVision-FeatureExtractor_CNN-Model-for-Cat-and-Dog-Prediction
# Insructions:
Tasks performed in the provided code:

1. **Data Preprocessing:**
   - Preprocessing the training set by uploading and extracting the dataset.
   - Preprocessing the test set by setting up the `ImageDataGenerator` and loading the test set.

2. **Building the CNN (Convolutional Neural Network):**
   - Initializing the CNN model using `Sequential`.
   - Adding convolutional layers with activation functions (ReLU) and pooling layers (MaxPool).
   - Adding a flattening layer and fully connected layers with ReLU activation.
   - Adding the output layer with a sigmoid activation function.

3. **Training the CNN:**
   - Compiling the CNN model with optimizer ('adam') and loss function ('binary_crossentropy').
   - Training the CNN model on the training set and evaluating its performance on the test set for 25 epochs.

4. **Making a Single Prediction:**
   - Loading a single image for prediction.
   - Preprocessing the image for the model.
   - Making a prediction using the trained CNN model.
   - Displaying the predicted class ('cat' or 'dog').

# Screenshots:
![Screenshot (11)](https://github.com/ArsalMirza007/DeepVision-FeatureExtractor_CNN-Model/assets/121928372/92037271-97e4-4d34-96b1-79ed1d40af73)
