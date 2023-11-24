import numpy as np


# 4x4 matrix
#                   class 1 predicted  class 2 predicted    class 3 predicted   class 4 predicted
#   class 1 actual         TP1                 E12                  E13              E14
#   class 2 actual         E21                 TP2                  E23              E24
#   class 3 actual         E31                 E32                  TP3              E34
#   class 4 actual         E41                 E42                  E43              TP4

# TPi is the true positives
# Eij is the false values where you want i but got j

# accuracy  = TP + TN / (TP + TN + FP + FN)
# precision = TP / (TP+FP)  -->  Precision of class 1 = TP1 / (TP1 + E21 + E31 + E41)
# Recall = TP/(TP+FN) --> recall of class 1 = TP1 / (TP1 + E12 + E13 + E14)
# F1 - measure = (2*precision*recall)/(precision + recall)

# Total test examples for any class = sum of corresponding row
# Total FN's for a class = sum of values in corresponding row (EXCLUDING the TP)
# Total FP's for a class = sum of values in corresponding column (EXCLUDING the TP)
# Total TN's for a certain class = sum of all rows and columns (excluding that class row and col)

def classification_metrics(confusion_matrix):
    # Accuracy calculation
    accuracy = np.sum(np.diagonal(confusion_matrix)) / np.sum(confusion_matrix)
    print("Accuracy: ", accuracy)
    class_count = confusion_matrix.shape[0]
    for i in range(class_count):
        # Precision calculation
        precision = confusion_matrix[i, i] / np.sum(confusion_matrix[:, i])
        # recall calculation
        recall = confusion_matrix[i, i] / np.sum(confusion_matrix[i, :])
        # f1_measure calculation
        f1_measure = (2 * precision * recall) / (precision + recall)
        print("\nclass: ", (i + 1), "\nprecision: ", precision, "\nrecall: ", recall, "\nf1 measure: ", f1_measure)
