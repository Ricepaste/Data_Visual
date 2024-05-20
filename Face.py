import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
import random as rd

from PCA import PCA
from LDA import LDA
import neural_network

DRAW = False
rd.seed(15)
np.random.seed(15)


def load_data():
    mat = sio.loadmat('ORL_RawData.mat')
    train_data = mat['ORLrawdataTrain']
    test_data = mat['ORLrawdataTest']
    ans = np.arange(1, 41)
    ans = np.repeat(ans, 5)

    return train_data, test_data, ans


def pca(train_data, test_data):
    pca = PCA(n_components=65)
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


def min_max_(pca_train_data, pca_test_data):
    min = np.min(pca_train_data, axis=0)
    max = np.max(pca_train_data, axis=0)
    minmax_train_data = (pca_train_data - min) / (max - min)
    minmax_test_data = (pca_test_data - min) / (max - min)
    print(f'shape of min-maxed training Data: {minmax_train_data.shape}')
    print(f'shape of min-maxed testing Data: {minmax_test_data.shape}')

    return minmax_train_data, minmax_test_data


def tnn(pca_train_data, pca_test_data, ans):
    # 進行隨機打亂
    idx = list(range(ans.size))
    rd.shuffle(idx)
    idx = np.array(idx)

    pca_train_data = pca_train_data[idx]
    pca_test_data = pca_test_data[idx]
    ans = ans[idx]

    # 歸一化
    minmax_train_data, minmax_test_data = min_max_(
        pca_train_data, pca_test_data)
    minmax_train_data = minmax_train_data * 0.99 + 0.01
    minmax_test_data = minmax_test_data * 0.99 + 0.01

    ans_onehot = np.zeros((ans.size, ans.max()))
    ans_onehot[np.arange(ans.size), (ans - 1).flatten()] = 1

    nn = neural_network.neuralNetwork(
        inputnodes=65, hiddennodes=150, outputnodes=40, lr=0.001)
    RMSE, AC = nn.train(minmax_train_data, ans_onehot, epochs=150)

    plt.plot(RMSE, color='r', marker='o',
             linewidth=2, markersize=6)
    plt.show()
    plt.plot(AC, color='r', marker='o',
             linewidth=2, markersize=6)
    plt.show()

    nn.query(minmax_test_data, ans_onehot)


def drawing(train_data, test_data, ans, pca_train_data, pca_test_data, X_projected, Y_projected):
    if (DRAW):
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

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(x1, x2, x3, c=ans, alpha=0.8,
                   cmap=plt.cm.get_cmap('rainbow', 200))
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')
        plt.show()

        print('shape of TestData:', test_data.shape)
        print('shape of PCA transformed TestData:', pca_test_data.shape)
        print('shape of LDA transformed TestData:', Y_projected.shape)

        tx1 = Y_projected[:, 0]
        tx2 = Y_projected[:, 1]
        tx3 = Y_projected[:, 2]

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.scatter(tx1, tx2, tx3, c=ans, alpha=0.8,
                   cmap=plt.cm.get_cmap('rainbow', 200))
        ax.set_xlabel('Principal Component 1')
        ax.set_ylabel('Principal Component 2')
        ax.set_zlabel('Principal Component 3')

        plt.show()
    else:
        pass


def ac_rate(X_projected, Y_projected, ans):
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
    ac_rate(X_projected, Y_projected, ans)
    tnn(pca_train_data, pca_test_data, ans)


if __name__ == "__main__":
    main()
