from neural_network import *
import scipy.io as sio

mat = sio.loadmat('ORL_PCA_MaxMin_Data.mat')

X_train = mat['TrainORL']
X_test = mat['TestORL']
Y = mat['Target']
Y_onehot = np.zeros((Y.size, Y.max()))
Y_onehot[np.arange(Y.size), (Y - 1).flatten()] = 1

n = neuralNetwork(inputnodes=65, hiddennodes=110, outputnodes=40, lr=0.0016)
Cross_Entropy, AC = n.train(X_train, Y_onehot, epochs=100)
plt.plot(Cross_Entropy, color='r', marker='o',
         linewidth=2, markersize=6)
plt.show()
plt.plot(AC, color='r', marker='o',
         linewidth=2, markersize=6)
plt.show()

n.query(X_test, Y_onehot)
