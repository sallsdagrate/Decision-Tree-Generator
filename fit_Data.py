from Decision_Tree.confusion_matrix import build_confusion_matrix
from Decision_Tree.metrics import classification_metrics
from Decision_Tree.decision_Tree_Generator import DecisionTreeGenerator
import numpy as np
from sys import argv


def main(dataset_path="wifi_db/clean_dataset.txt", show_plots=False):
    n_classes = 4  # Change this to the actual number of classes
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=int)

    with open(dataset_path, 'r') as db:
        data = np.loadtxt(db)

    gen = DecisionTreeGenerator(data)

    # will generate all different combinations from the data
    trees_list = gen.n_fold_cross_val_split(10, show_plots)

    all_predictions = []
    all_actuals = []

    # also output accuracy
    # Make test set
    for tree, depth, test_set in trees_list:
        all_predictions.extend(tree.get_predictions(test_set))
        all_actuals.extend(int(value[7]) for value in test_set)

    # Calculate the confusion matrix for the current fold
    fold_confusion_matrix = build_confusion_matrix(all_predictions, all_actuals, n_classes)

    # Add the counts from the fold's confusion matrix to the overall confusion matrix
    confusion_matrix += fold_confusion_matrix

    # Divide each count by the number of folds to get the average
    average_confusion_matrix = confusion_matrix / len(trees_list)

    print("Averaged Confusion Matrix:")
    print(average_confusion_matrix)
    classification_metrics(average_confusion_matrix)


if __name__ == "__main__":
    if len(argv) > 1:
        sd = False
        fp = "wifi_db/clean_dataset.txt"
        for a in argv:
            if a.startswith("--save_diagrams=") and a.endswith("True"):
                sd = True
            elif a.startswith("--dataset="):
                fp = a.removeprefix("--dataset=")
        main(fp, sd)
    else:
        main()
