# viridis inferno rainbow

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA
from LDA import LDA
import neural_network

DRAW = True


def load_data():
    mat = sio.loadmat('ORL_RawData.mat')
    train_data = mat['ORLrawdataTrain']
    test_data = mat['ORLrawdataTest']
    ans = np.arange(1, 41)
    ans = np.repeat(ans, 5)

    return train_data, test_data, ans


def pca(train_data, test_data):
    pca = PCA(n_components=50)
    pca.fit(train_data)
    pca_train_data = pca.project(train_data).real
    pca_test_data = pca.project(test_data).real

    return pca_train_data, pca_test_data


def lda(pca_train_data, pca_test_data, ans):
    lda = LDA(20)
    lda.fit(pca_train_data, ans)
    X_projected = lda.transform(pca_train_data)
    Y_projected = lda.transform(pca_test_data)

    return X_projected, Y_projected


def drawing(train_data, test_data, ans, pca_train_data, pca_test_data, X_projected, Y_projected):
    print('shape of TrainData:', train_data.shape)
    print('shape of PCA transformed TrainData:', pca_train_data.shape)
    print('shape of LDA transformed TrainData:', X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]
    x3 = X_projected[:, 2]

    plt.scatter(x1, x2,
                c=ans, edgecolor='none', alpha=0.8,
                cmap=plt.cm.get_cmap('rainbow', 200))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar()
    plt.show()

    if (DRAW):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.scatter(x1, x2, x3, c=ans, alpha=0.8,
                   cmap=plt.cm.get_cmap('rainbow', 200))
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend()
        plt.show()

    print('shape of TestData:', test_data.shape)
    print('shape of PCA transformed TestData:', pca_test_data.shape)
    print('shape of LDA transformed TestData:', Y_projected.shape)

    tx1 = Y_projected[:, 0]
    tx2 = Y_projected[:, 1]
    tx3 = Y_projected[:, 2]

    if (DRAW):
        fig = plt.figure()
        ax = fig.gca(projection='3d')

        ax.scatter(tx1, tx2, tx3, c=ans, alpha=0.8,
                   cmap=plt.cm.get_cmap('rainbow', 200))
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        ax.legend()

        plt.show()


def ac_rate(X_projected, Y_projected):
    mean = []
    for i in range(40):
        mean.append(np.mean(X_projected[i*5:i*5+5], axis=0))

    success = 0
    three = 0

    for i in range(200):
        min = float('inf')
        dis = []
        min_class = None
        for j in range(40):
            euclidean_distance = np.linalg.norm(Y_projected[i] - mean[j])

            dis.append(euclidean_distance)

            if (euclidean_distance < min):
                min = euclidean_distance
                min_class = j
        dis = np.array(dis)
        dis = np.argsort(dis)
        if (min_class == i // 5):
            success += 1
        if (dis[0] == i // 5 or dis[1] == i // 5 or dis[2] == i // 5):
            three += 1
    success = success / 200.0 * 100.
    three = three / 200.0 * 100.
    print("success rate = {}%".format(success))
    print("top three rate = {}%".format(three))


def main():
    train_data, test_data, ans = load_data()
    pca_train_data, pca_test_data = pca(train_data, test_data)
    X_projected, Y_projected = lda(pca_train_data, pca_test_data, ans)
    drawing(train_data, test_data, ans, pca_train_data,
            pca_test_data, X_projected, Y_projected)


if __name__ == "__main__":
    main()
