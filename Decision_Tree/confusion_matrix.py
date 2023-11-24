import numpy as np


# initialises n x n matrix with zeroes.
# Compares prediction and actual and inserts into correct position
def build_confusion_matrix(predictions, actuals, n_classes):
    conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for pred, act in zip(predictions, actuals):
        conf_matrix[act - 1][pred - 1] += 1  # Change if labels start from 1,2,3,4
    return conf_matrix
