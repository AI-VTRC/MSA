from sklearn.decomposition import (
    PCA,
    TruncatedSVD,
)
import matplotlib.pyplot as plt
import matplotlib
import numpy as np


def plot_LSA(test_data, test_labels, plot=True):
    lsa = TruncatedSVD(n_components=2)
    lsa.fit(test_data)
    lsa_scores = lsa.transform(test_data)
    color_mapper = {label: idx for idx, label in enumerate(set(test_labels))}
    color_column = [color_mapper[label] for label in test_labels]
    colors = ["orange", "blue", "red", "green"]
    if plot:
        plt.scatter(
            lsa_scores[:, 0],
            lsa_scores[:, 1],
            s=8,
            alpha=0.8,
            c=test_labels,
            cmap=matplotlib.colors.ListedColormap(colors),
        )
        # plt.legend(handles=[orange_patch, blue_patch], prop={'size': 20})


def svd(x, y, xS, yS):
    x_train = np.array([image.flatten() for image in x])
    x_trainS = np.array([image.flatten() for image in xS])

    del x, xS

    fig = plt.figure(figsize=(20, 10))
    fig.add_subplot(121)
    plot_LSA(x_train, y)
    plt.title("pre_SMOTE")
    fig.add_subplot(122)
    plot_LSA(x_trainS, yS)
    plt.title("post_SMOTE")
    plt.savefig("../pathology_dataset/results/svd_result.png")
