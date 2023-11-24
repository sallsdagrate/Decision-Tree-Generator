import os.path
from math import log2, floor
from sys import argv
from time import process_time

import numpy as np

from Decision_Tree.decision_Tree import DecisionTreeDecorator

# GLOBAL VARS
ROOM_IDX = 7
NO_OF_FEATURES = 8
FOLDS = 10

SHOW_PLOTS = True


class DecisionTreeGenerator:
    def __init__(self, data: np.array):
        self.data = data
        self.data_len = len(data)

    # finds the optimum feature value to split at based on a passed in dataset
    #
    # This algorithm works linearly, as opposed to the less efficient but more
    # verbose version using numpy array splicing which we implemented previously.
    # According to our measurements, this algorithm is faster by an order of magnitude.
    def __find_split__(self, data_to_split: np.array = None) -> (int, int):
        if data_to_split is None:
            data_to_split = self.data

        # variables to keep track of best entropy and corresponding split
        res, val, ft = float('inf'), 0, None
        # loop through each feature
        for feature in range(ROOM_IDX):
            # sort by feature and discard other values
            d = data_to_split[data_to_split[:, feature].argsort(kind='mergesort')][:, [feature, ROOM_IDX]]

            # get a set of feature values and use index to group together outputs from same feature value
            unique, index = np.unique(d[:, 0], return_index=True)
            splits = np.split(d[:, 1], index[1:])

            # counts is in form [[ai1...ai4]] where aij is the number of feature value i with output j
            counts = [[np.count_nonzero(x == i + 1) for i in range(4)] for x in splits]

            # calculate sums to find how many j outputs in left partition
            cumulatives = np.cumsum(counts, axis=0)

            # calulate how many items in left partion being the sum of number of j outputs
            sums = np.sum(cumulatives, axis=1)
            total = sums[-1]

            # go through possible splits
            for i in range(0, len(cumulatives) - 1):

                # compute entropy on the left and right side of partition at unique[i]
                el, er = 0, 0
                for j in range(4):

                    # find proportions on the left and right partitions
                    pl = cumulatives[i][j] / sums[i]
                    pr = (cumulatives[-1][j] - cumulatives[i][j]) / (total - sums[i])
                    if pl != 0:
                        el -= pl * log2(pl)
                    if pr != 0:
                        er -= pr * log2(pr)

                # find remainder
                e = el * sums[i] / total + er * (1 - sums[i] / total)

                # set results if better entropy found
                if e <= res:
                    res = e
                    val = (unique[i] + unique[i + 1]) / 2
                    ft = feature

        # return feature and value of split
        return ft, val

    # decision tree learning algorithm provided in the spec
    def __decision_tree_learning__(self, depth: int, training_data=None) -> (dict, int):
        if training_data is None:
            training_data = self.data

        # if all labels have the same value
        if np.all(training_data == training_data[0, :], axis=0)[ROOM_IDX]:
            # return leaf node
            return (
                {
                    'leaf': True,
                    'label': int(training_data[0, ROOM_IDX])
                },
                depth
            )
        else:
            # otherwise find optimal split and create left and right decision trees
            split_feature, split_value = self.__find_split__(training_data)
            l_tree, l_depth = self.__decision_tree_learning__(
                depth + 1,
                training_data[
                    training_data[:, split_feature] <= split_value
                    ])
            r_tree, r_depth = self.__decision_tree_learning__(
                depth + 1,
                training_data[
                    training_data[:, split_feature] > split_value
                    ])
            # return child node
            return (
                {
                    'leaf': False,
                    'split_feature': split_feature,
                    'split_value': split_value,
                    'left': l_tree,
                    'right': r_tree
                },
                max(l_depth, r_depth)
            )

    # returns the training and testing data based on the current fold and the
    # required size of the training data
    def __split_data__(self, test_data_size, j):
        lb, ub = j * test_data_size, (j + 1) * test_data_size
        r = range(lb, ub)
        test = self.data[[x for x in r]]
        train = self.data[[x for x in range(self.data_len) if x not in r]]
        print(f'test data range: {lb}, {ub}')
        return test, train

    # takes a tree and a sample and returns the predicted label for this sample
    def n_fold_cross_val_split(self, n, show_plots=SHOW_PLOTS) -> list:
        trees_list = []
        test_data_size = floor(self.data_len / n)

        for i in range(n):
            print(f'FOLD {i + 1}')
            ft = process_time()

            test_data, train_data = self.__split_data__(test_data_size, i)

            tree, depth = self.__decision_tree_learning__(0, train_data)
            d_tree = DecisionTreeDecorator(tree)
            print(f'trained tree of depth {depth}')
            print(f'TIME ELAPSED FOLD {i + 1} {process_time() - ft}s')
            print(f'------------------------------------------------')
            if show_plots:
                d_tree.plot_tree(i + 1)
            trees_list.append((d_tree, depth, test_data))

        return trees_list


def main(dataset_path, folds):
    with open(dataset_path, 'r') as db:
        data = np.loadtxt(db)

    t = process_time()

    gen = DecisionTreeGenerator(data)
    _ = gen.n_fold_cross_val_split(folds)

    total_elapsed = process_time() - t
    print(f'TIME ELAPSED FOR {folds} FOLD TESTING AND TRAINING {total_elapsed}s')
    print(f'AVERAGE TIME ELAPSED PER FOLD {total_elapsed / folds}s')


if __name__ == "__main__":
    no_args = len(argv)

    if no_args < 2:
        exit()

    fp = argv[1]
    print(fp)
    if not (fp.endswith(".txt") and os.path.exists(fp)):
        print('error with dataset file path')
        exit()

    fs = FOLDS
    if no_args == 3:
        if int(argv[2]) > 0:
            fs = int(argv[2])

    SHOW_PLOTS = False if no_args == 4 and argv[3].lower() == "false" else True

    main(fp, fs)
