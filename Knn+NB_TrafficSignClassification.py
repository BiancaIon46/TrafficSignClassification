import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
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

def show_results(images, labels, predictions, class_probabilities):
    for i in range(len(images)):
        plt.imshow(images[i].reshape((200, 200, 3)))
        true_label = labels[i]
        predicted_label = predictions[i]
        probabilities = class_probabilities[i]

        plt.title(f'True Label: {true_label}, Predicted Label: {predicted_label}')

        # Display the likelihood percentage for each class
        for j, prob in enumerate(probabilities):
            plt.text(210, j * 40 + 20, f'{class_names[j]}: {prob * 100:.2f}%', fontsize=10, color='red')

        plt.show()

def plot_results(y_test, predictions, class_names):
    cm = confusion_matrix(y_test, predictions, labels=class_names)
    plt.figure(figsize=(8, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (NB on KNN)')
    plt.colorbar()

    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    plt.show()

dataset_folder = 'D:\\Bianca\\acs\\an3\\sem1\\TIA\\lab\\Teme\\dataset'
X, y = load_images_and_labels(dataset_folder)

# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Train K-Nearest Neighbors
knn_clf = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='ball_tree', leaf_size=20, p=2)
knn_clf.fit(X_train, y_train)

# Predictions for validation set using KNN
val_predictions_knn = knn_clf.predict(X_val)

# Accuracy for validation set using KNN
val_accuracy_knn = accuracy_score(y_val, val_predictions_knn)
print(f'Validation Accuracy (KNN): {val_accuracy_knn * 100:.2f}%')

# Predictions for test set using KNN
test_predictions_knn = knn_clf.predict(X_test)

# Accuracy for test set using KNN
test_accuracy_knn = accuracy_score(y_test, test_predictions_knn)
print(f'Test Accuracy (KNN): {test_accuracy_knn * 100:.2f}%')

# Naive Bayes on the predictions made by KNN
nb_clf = GaussianNB(priors=None, var_smoothing=1e-9)
nb_clf.fit(X_test, test_predictions_knn)

# Predictions for test set using Naive Bayes
test_predictions_nb = nb_clf.predict(X_test)

# Accuracy for test set using Naive Bayes
test_accuracy_nb = accuracy_score(y_test, test_predictions_nb)
print(f'Test Accuracy (Naive Bayes on KNN Predictions): {test_accuracy_nb * 100:.2f}%')

class_names = np.unique(y_test)
print("Classification Report for Test Set (Naive Bayes on KNN Predictions):\n", classification_report(y_test, test_predictions_nb))
print("Confusion Matrix for Test Set (Naive Bayes on KNN Predictions):\n", confusion_matrix(y_test, test_predictions_nb))

show_results(X_test, y_test, test_predictions_nb, nb_clf.predict_proba(X_test))
plot_results(y_test, test_predictions_nb, class_names)
