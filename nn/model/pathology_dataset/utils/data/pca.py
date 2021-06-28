from sklearn.decomposition import (
    PCA,
)  # Principal component analysis (PCA), dimensionality reduction using truncated SVD

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def pca(x, y, xS, yS):
    x_train = np.array([image.flatten() for image in x])
    x_trainS = np.array([image.flatten() for image in xS])

    del x, xS

    pca = PCA()
    pca.fit(x_trainS)

    U = pca.transform(x_trainS)
    S = pca.explained_variance_
    V = pca.components_

    print("U.shape = ", U.shape)
    print("S.shape = ", S.shape)
    print("V.shape = ", V.shape)

    plt.rc("image", cmap="binary")
    plt.figure(figsize=(8, 5))
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(
            tf.keras.preprocessing.image.array_to_img(x_train[i].reshape(200, 200, 3))
        )
        plt.title(y[i])
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.savefig("../pathology_dataset/results/pca_train_img.png")

    print("plot the first principal components")

    plt.figure(figsize=(8, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(V[i].reshape(200, 200, 3)))
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.savefig("../pathology_dataset/results/pca_1.png")

    print("plot less relevant principal components")
    plt.figure(figsize=(8, 5))
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.imshow(
            tf.keras.preprocessing.image.array_to_img(V[200 + i].reshape(200, 200, 3))
        )
        plt.xticks(())
        plt.yticks(())
    plt.tight_layout()
    plt.savefig("../pathology_dataset/results/pca_2.png")

    # Here we plot the explained variance as a function of the principal directions retained.
    ev_cumsum = np.cumsum(pca.explained_variance_) / (pca.explained_variance_).sum()
    ev_at90 = ev_cumsum[ev_cumsum < 0.9].shape[0]
    print(ev_at90)

    plt.plot(ev_cumsum)
    plt.title("Explained Variance")
    plt.xlabel("Components")
    plt.xticks([0, ev_at90, 1000, 1500, 2000, 2500])
    plt.yticks(list(plt.yticks()[0]) + [0.9])
    plt.vlines(ev_at90, 0, 1, linestyles="dashed")
    plt.hlines(0.9, 0, 1000, linestyles="dashed")
    plt.savefig("../pathology_dataset/results/pca_explained_variance.png")
