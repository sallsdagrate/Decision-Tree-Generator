import collections
import os.path

import numpy as np
import matplotlib.pyplot as plt


class DecisionTreeDecorator:

    def __init__(self, tree: dict):
        self.tree = tree

    # generates a matplotlib plot from root downwards
    def __plot_root__(self, root=None, pos=(0., 0.), dx=10., dy=20., ax=None):
        if root is None:
            root = self.tree

        # Initialise graph
        if ax is None:
            fig, ax = plt.subplots()
            fig.set_figwidth(15)
            fig.set_figheight(9)

        def __plot_leaf__(x, y, label):
            ax.text(x, y,
                    f"{label}",
                    ha='center', va='center',
                    fontsize='medium',
                    bbox=dict(
                        boxstyle='circle',
                        facecolor='w',
                        edgecolor='r'
                    ))

        def __plot_node__(x, y, feature, value):
            ax.text(x, y,
                    f"X{feature} < {value}",
                    ha='center',
                    va='center',
                    fontsize='x-small',
                    bbox=dict(
                        boxstyle='Round',
                        facecolor='w',
                        edgecolor='b'
                    ))

        if root['leaf']:
            __plot_leaf__(pos[0], pos[1], root['label'])
            return

        __plot_node__(pos[0], pos[1], root['split_feature'], root['split_value'])

        # queue in the children to prepare breadth-first-traversal
        q = collections.deque([(root['left'], pos[0]), (root['right'], pos[0])])
        # setting w to be max width of graph
        w, y = 2 * dx, pos[1]
        ax.set_xlim(-dx, dx)
        while q:
            # x to be evenly spaced points along the width
            div = len(q)
            x = (w / div) / 2 - dx
            y -= dy
            # every item in the queue right now is from the same generation and belongs on the same level
            for i in range(div):
                ax.plot(x, y)
                elem, px = q.popleft()
                if elem['leaf']:
                    __plot_leaf__(x, y, elem['label'])
                else:
                    __plot_node__(x, y, elem['split_feature'], elem['split_value'])
                    # add children to the queue
                    q.append((elem['left'], x))
                    q.append((elem['right'], x))
                ax.plot([px, x], [y + dy, y], 'b-')
                # compute next evenly spaced position
                x += (w / div)

    # returns the predicted value of a sample based on the decision trained tree
    def evaluate_sample(self, sample: np.array, subtree: dict = None) -> int:
        if subtree is None:
            subtree = self.tree

        if subtree['leaf']:
            return subtree['label']
        else:
            feature = subtree['split_feature']
            value = subtree['split_value']

            if sample[feature] <= value:
                return self.evaluate_sample(sample, subtree['left'])
            else:
                return self.evaluate_sample(sample, subtree['right'])

    # calculates the predictions of the test data samples based on the trained decision tree.
    def get_predictions(self, test_data):
        return [self.evaluate_sample(sample) for sample in test_data]

    # generated a tree visualisation and saves to a file with file name based on the value of n passed in
    def plot_tree(self, n):
        self.__plot_root__()
        plt.axis('off')
        if not os.path.exists("trees"):
            os.makedirs("trees")
        plt.savefig(f'trees/fold_{n}_tree.png')
