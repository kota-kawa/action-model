import cv2
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

# Create a callback for early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5)

# Dataset path
dataset_path = "faces_dataset"

import tensorflow as tf

# Set random seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

# Function to load images and add them to a list along with their labels
# num_augmentations=5 specifies the number of augmented images
def load_images_and_labels(dataset_path, target_size=(100, 100), augment=True, num_augmentations=15):
    images = []
    labels = []
    label_id = 0

    # Read images from each folder
    for person_folder in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_folder)
        if os.path.isdir(person_path):
            for image_file in os.listdir(person_path):
                image_path = os.path.join(person_path, image_file)
                # Load the image in grayscale
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # Detect faces and get coordinates
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                    if len(faces) > 0:
                        # If a face is detected, get the largest face
                        faces = sorted(faces, key=lambda x: x[2]*x[3], reverse=True)
                        (x, y, w, h) = faces[0]
                        # Crop and resize the face
                        face_img = image[y:y+h, x:x+w]
                        face_img = cv2.resize(face_img, target_size)

                        if augment:
                            # Data augmentation
                            for _ in range(num_augmentations):
                                augmented_img = augment_image(face_img)
                                images.append(augmented_img)
                                labels.append(label_id)
            label_id += 1

    return images, labels

# Data augmentation for images
def augment_image(image):
    # Rotation
    angle = random.uniform(-20, 20)
    rows, cols = image.shape
    rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix, (cols, rows))
    # Translation
    x_translation = random.uniform(-10, 10)
    y_translation = random.uniform(-10, 10)
    translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation]])
    image = cv2.warpAffine(image, translation_matrix, (cols, rows))
    # Flip
    flip = random.choice([True, False])
    if flip:
        image = cv2.flip(image, 1)
    # Brightness change
    brightness = random.uniform(0.5, 2.0)
    image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)
    # Sharpness change
    if random.choice([True, False]):
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        image = cv2.filter2D(image, -1, kernel)
    # Shear transformation
    shear = random.uniform(-0.3, 0.3)
    shear_matrix = np.float32([[1, shear, 0], [0, 1, 0]])
    image = cv2.warpAffine(image, shear_matrix, (cols, rows))
    # Zoom
    zoom = random.uniform(0.8, 1.2)
    if zoom > 1.0:
        # Zoom in
        image = cv2.resize(image, None, fx=zoom, fy=zoom)
        # Resize the image again after zooming
        image = cv2.resize(image, (100, 100))
    else:
        # Zoom out (wide angle)
        image = cv2.resize(image, None, fx=zoom, fy=zoom)
        # Resize the image again after zooming
        image = cv2.copyMakeBorder(image, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT, value=[0,0,0])
        image = cv2.resize(image, (100, 100))

    # Color change
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image[:,:,0] = image[:,:,0] + random.uniform(-10, 10)
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image

# Prepare the data
images, labels = load_images_and_labels(dataset_path)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Convert data to NumPy arrays
X_train = np.array(X_train)
X_test = np.array(X_test)

# One-hot encode the labels
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Normalize the data
X_train = X_train / 255.0
X_test = X_test / 255.0

# Change data dimensions (add color channels)
X_train = np.stack((X_train,)*3, axis=-1)
X_test = np.stack((X_test,)*3, axis=-1)

# Load the VGG16 model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(100, 100, 3))

# Add new fully connected layers
x = base_model.output
x = Flatten()(x)
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)  # Increase neurons and add L2 regularization
x = Dropout(0.5)(x)  # Add dropout
x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)  # Increase neurons and add L2 regularization
x = Dropout(0.5)(x)  # Add dropout
predictions = Dense(len(np.unique(labels)), activation='softmax')(x)

# Create a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the first 15 layers (fine-tune more layers)
for layer in model.layers[:15]:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Save the trained model
model.save('face_model.h5')

# Evaluate the model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print("Accuracy:", accuracy)
