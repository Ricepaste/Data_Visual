# viridis inferno rainbow

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from PCA import PCA
from LDA import LDA


mat = sio.loadmat('ORL_RawData.mat')
# print(mat)

draw = True

train_data = mat['ORLrawdataTrain']
pos = np.arange(1, 41)
pos = np.repeat(pos, 5)
# print(pos)

# print(type(train_data))
# print(train_data.shape)
# print(pos)

# test = np.array([[1, 3], [2, 4]])
# print(np.mean(test, axis=0))
pca = PCA(0.8)
pca.fit(train_data)
print(train_data.shape)
pca_data = pca.project(train_data).real
print(pca_data.shape)

lda = LDA(30)
lda.fit(pca_data, pos)
X_projected = lda.transform(pca_data)

print('shape of TrainData:', train_data.shape)
print('shape of PCA transformed TrainData:', pca_data.shape)
print('shape of LDA transformed TrainData:', X_projected.shape)

x1 = X_projected[:, 0]
x2 = X_projected[:, 1]
x3 = X_projected[:, 2]


# x1 = x1[:10]
# x2 = x2[:10]
# x3 = x3[:10]
# pos = pos[:10]

plt.scatter(x1, x2,
            c=pos, edgecolor='none', alpha=0.8,
            cmap=plt.cm.get_cmap('rainbow', 200))
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.show()

if (draw):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(x1, x2, x3, c=pos, alpha=0.8,
               cmap=plt.cm.get_cmap('rainbow', 200))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

    # print(x1)
    # print(x2)
    # print(x3)

    plt.show()

# --------------------------------------------------------------------------------------------


test_data = mat['ORLrawdataTest']
pos = np.arange(1, 41)
pos = np.repeat(pos, 5)

pca_data = pca.transform(test_data).real
Y_projected = lda.transform(pca_data)

print('shape of TestData:', test_data.shape)
print('shape of PCA transformed TestData:', pca_data.shape)
print('shape of LDA transformed TestData:', Y_projected.shape)

tx1 = Y_projected[:, 0]
tx2 = Y_projected[:, 1]
tx3 = Y_projected[:, 2]


# tx1 = tx1[:20]
# tx2 = tx2[:20]
# tx3 = tx3[:20]
# pos = pos[:20]

if (draw):
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(tx1, tx2, tx3, c=pos, alpha=0.8,
               cmap=plt.cm.get_cmap('rainbow', 200))
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

    # print(x1)
    # print(x2)
    # print(x3)

    plt.show()

# print(X_projected.shape)
# print(Y_projected.shape)
mean = []
for i in range(40):
    mean.append(np.mean(X_projected[i*5:i*5+5], axis=0))
# print(len(mean))

success = 0
three = 0

for i in range(200):
    min = 99999999999999
    dis = []
    min_class = None
    for j in range(40):
        euclidean_distance = np.linalg.norm(Y_projected[i] - mean[j])

        dis.append(euclidean_distance)

        if (euclidean_distance < min):
            min = euclidean_distance
            min_class = j
    # print(len(dis))
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
