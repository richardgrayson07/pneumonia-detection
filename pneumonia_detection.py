import tensorflow as tf

from tensorflow import keras



# Check the version of keras to confirm the import
print(keras.__version__)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),  # Specify input shape
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print model summary
model.summary()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Rescaling pixel values to [0, 1]
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
   r"C:\Users\sarth\Downloads\chest_xray\train",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
val_generator = val_datagen.flow_from_directory(
     r"C:\Users\sarth\Downloads\chest_xray\val",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
test_generator = test_datagen.flow_from_directory(
    r"C:\Users\sarth\Downloads\chest_xray\test",
    target_size=(64, 64),
    batch_size=32,
    class_mode='binary'
)
# Train the model
history = model.fit(
    train_generator,  # Training data
    steps_per_epoch=len(train_generator),  # Number of batches per epoch
    validation_data=val_generator,  # Validation data
    validation_steps=len(val_generator),  # Number of batches for validation
    epochs=10  # Number of epochs (adjust as needed)
)
# Evaluate the model on test data
test_loss, test_accuracy = model.evaluate(test_generator)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
import matplotlib.pyplot as plt

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
 
model.save("pneumonia_detection_model.keras")
print("Model saved!")


from tensorflow.keras.models import load_model
model = load_model("pneumonia_detection_model.h5")

import numpy as np
from tensorflow.keras.preprocessing import image

# Load an image
img_path = ""  # Replace with your image path
img = image.load_img(img_path, target_size=(64, 64))  # Resize to match input shape
img_array = image.img_to_array(img) / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict
prediction = model.predict(img_array)
if prediction[0] > 0.5:
    print("Prediction: PNEUMONIA")
else:
    print("Prediction: NORMAL")


