import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

def load_images_and_labels(folder):
    images = []
    labels = []
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                image_path = os.path.join(label_path, file)
                img = cv2.imread(image_path)
                img = cv2.resize(img, (200, 200))  # Resize images to a consistent size
                images.append(img.flatten())  # Flatten the image array
                labels.append(label)
    return np.array(images), np.array(labels)

def show_results(images, labels, predictions):
    for i in range(len(images)):
        plt.imshow(images[i].reshape((200, 200, 3)))
        plt.title(f'Actual: {labels[i]}, Predicted: {predictions[i]}')
        plt.show()

def plot_results(y_test, predictions, class_names):
    cm = confusion_matrix(y_test, predictions, labels=class_names)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.show()

dataset_folder = 'D:\\Bianca\\acs\\an3\\sem1\\TIA\\lab\\Tema1\\dataset'
X, y = load_images_and_labels(dataset_folder)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train
clf = SVC(kernel='linear', C=1.0)
clf.fit(X_train, y_train)

# Predictions for validation set
val_predictions = clf.predict(X_val)

# Accuracy for validation set
val_accuracy = accuracy_score(y_val, val_predictions)
print(f'Validation Accuracy: {val_accuracy * 100:.2f}%')

# Predictions for test set
test_predictions = clf.predict(X_test)

# Accuracy for test set
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

show_results(X_test, y_test, test_predictions)

class_names = np.unique(y_test)
print("Classification Report for Test Set:\n", classification_report(y_test, test_predictions))
print("Confusion Matrix for Test Set:\n", confusion_matrix(y_test, test_predictions))

plot_results(y_test, test_predictions, class_names)
