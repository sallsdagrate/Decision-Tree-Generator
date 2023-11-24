import numpy as np
import os

file_path = '../wifi_db/clean_dataset.txt'
load_file = np.loadtxt(file_path)


def read_dataset(filepath):
    feature_vectors = []
    class_labels = []

    # Open the file and process its lines
    for line in open(filepath):
        # Check if the line is not empty
        if line.strip():
            # Split the line into parts
            parts = line.strip().split()
            print(parts)
            # print(f"parts{parts}")
            # Extract feature values (convert to float) and class label
            features = list(map(float, parts[:-1]))
            # print(f"features{features}")
            label = parts[-1]

            # Append the feature vector and class label to the respective lists
            feature_vectors.append(features)
            class_labels.append(label)

    # Encode class labels and get unique classes
    classes, encoded_labels = np.unique(class_labels, return_inverse=True)

    # Convert feature vectors and encoded labels to NumPy arrays
    feature_vectors = np.array(feature_vectors)
    encoded_labels = np.array(encoded_labels)

    # Return the processed data and unique classes
    return feature_vectors, encoded_labels, classes


(x, y, classes) = read_dataset(file_path)
print(x.shape)
print(y.shape)
print(classes)
