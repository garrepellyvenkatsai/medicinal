import json
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import MobileNet, ResNet50, VGG16
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import pickle

# Paths to the new directory structure
dataset_dir = 'D:/medicinal_plant_project/dataset/all_data'  # Path to dataset
model_save_path = 'D:/medicinal_plant_project/models/medicinal_leaf_classifier.h5'  # Path to save model
class_indices_path = 'D:/medicinal_plant_project/models/class_indices.json'  # Path to save class indices
plant_info_path = 'D:/medicinal_plant_project/models/plant_info.json'  # Path to plant info JSON
results_path = 'D:/medicinal_plant_project/models/results.json'  # Path to save comparison results

# Constants
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 3

# Load plant information from JSON file
try:
    with open(plant_info_path, 'r') as f:
        plant_info = json.load(f)
    print(f"Plant information loaded from {plant_info_path}")
except Exception as e:
    print(f"Error loading plant information: {e}")

# Load all images and their labels from the dataset
images = []
labels = []

# Iterate through all plant categories (labels) in the dataset
for label in os.listdir(dataset_dir):
    label_dir = os.path.join(dataset_dir, label)
    if os.path.isdir(label_dir):
        print(f"Processing label directory: {label_dir}")  # Debugging line
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)
            if img_name.endswith(('.png', '.jpg', '.jpeg')):  # Check for valid image formats
                try:
                    print(f"Loading image: {img_path}")  # Debugging line
                    img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")  # Error handling
            else:
                print(f"Skipping non-image file: {img_path}")  # Skipping non-image files

# Check if images were loaded
print(f"Total images loaded: {len(images)}")
if len(images) == 0:
    print("No images found. Please check your dataset path.")

# Convert lists to numpy arrays
images = np.array(images) / 255.0  # Normalize images to [0, 1]
labels = np.array(labels)

# Map labels to integers
class_names = list(set(labels))
class_indices = {class_name: i for i, class_name in enumerate(class_names)}
labels = np.array([class_indices[label] for label in labels])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Model Training Functions
def train_deep_learning_models():
    # MobileNet Model
    base_model = MobileNet(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False
    model_mobilenet = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(len(class_indices), activation='softmax')
    ])
    model_mobilenet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_mobilenet.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    mobilenet_accuracy = model_mobilenet.evaluate(X_test, y_test, verbose=0)[1]

    # ResNet Model
    base_model_resnet = ResNet50(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model_resnet.trainable = False
    model_resnet = Sequential([
        base_model_resnet,
        GlobalAveragePooling2D(),
        Dense(len(class_indices), activation='softmax')
    ])
    model_resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_resnet.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    resnet_accuracy = model_resnet.evaluate(X_test, y_test, verbose=0)[1]

    # VGG16 Model
    base_model_vgg = VGG16(input_shape=(IMG_SIZE, IMG_SIZE, 3), include_top=False, weights='imagenet')
    base_model_vgg.trainable = False
    model_vgg = Sequential([
        base_model_vgg,
        GlobalAveragePooling2D(),
        Dense(len(class_indices), activation='softmax')
    ])
    model_vgg.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model_vgg.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=BATCH_SIZE)
    vgg_accuracy = model_vgg.evaluate(X_test, y_test, verbose=0)[1]

    return mobilenet_accuracy, resnet_accuracy, vgg_accuracy

def train_machine_learning_models():
    # Flatten images for ML models
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_flat, y_train)
    rf_pred = rf.predict(X_test_flat)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    # Train SVM Classifier
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(X_train_flat, y_train)
    svm_pred = svm.predict(X_test_flat)
    svm_accuracy = accuracy_score(y_test, svm_pred)

    return rf_accuracy, svm_accuracy

# Compare models
mobilenet_acc, resnet_acc, vgg_acc = train_deep_learning_models()
rf_acc, svm_acc = train_machine_learning_models()

# Store the results
results = {
    'MobileNet': mobilenet_acc,
    'ResNet50': resnet_acc,
    'VGG16': vgg_acc,
    'Random Forest': rf_acc,
    'SVM': svm_acc
}

with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

# Save class indices for future use (like predictions)
with open(class_indices_path, 'w') as f:
    json.dump(class_indices, f, indent=4)
print(f"Class indices saved to '{class_indices_path}'")

# Function to get plant information from the JSON file
def get_plant_info(plant_name):
    if plant_name in plant_info:
        return plant_info[plant_name]
    else:
        return {'uses': 'Information not available', 'benefits': 'Information not available', 'locations': 'Information not available'}

# Function to make a prediction and return plant info
def predict_plant(image_path):
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Make prediction
    prediction = model_mobilenet.predict(img_array)
    predicted_class_idx = np.argmax(prediction)
    
    # Get plant name from class index
    plant_name = list(class_indices.keys())[list(class_indices.values()).index(predicted_class_idx)]

    # Retrieve and display plant information
    plant_details = get_plant_info(plant_name)
    print(f"Predicted Plant: {plant_name}")
    print(f"Uses: {plant_details['uses']}")
    print(f"Benefits: {plant_details['benefits']}")
    print(f"Locations: {plant_details['locations']}")

# Example usage:
image_path = 'D:/medicinal_plant_project/dataset/test/Aloevera/28.jpg'  # Replace with actual path to an image
predict_plant(image_path)
