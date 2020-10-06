import random
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split as ttsplit
import sys
sys.path.append('mnist')
import mnist

# 损失函数
class LossFun:
    def __init__(self, lf_type="least_square"):
        self.name = "loss function"
        self.type = lf_type

    def cal(self, t, z):
        loss = 0
        if self.type == "least_square":
            loss = self.least_square(t, z)
        return loss

    def cal_deriv(self, t, z):
        delta = 0
        if self.type == "least_square":
            delta = self.least_square_deriv(t, z)
        return delta

    def least_square(self, t, z):
        zsize = z.shape
        sample_num = zsize[1]
        return np.sum(0.5 * (t - z) * (t - z) * t) / sample_num

    def least_square_deriv(self, t, z):
        return z - t

# 激活函数
class ActivationFun:
    def __init__(self, atype="sigmoid"):
        self.name = "activation function library"
        self.type = atype

    def cal(self, a):
        z = 0
        if self.type == "sigmoid":
            z = self.sigmoid(a)
        elif self.type == "relu":
            z = self.relu(a)
        return z

    def cal_deriv(self, a):
        z = 0
        if self.type == "sigmoid":
            z = self.sigmoid_deriv(a)
        elif self.type == "relu":
            z = self.relu_deriv(a)
        return z

    def sigmoid(self, a):
        return 1 / (1 + np.exp(-a))

    def sigmoid_deriv(self, a):
        fa = self.sigmoid(a)
        return fa * (1 - fa)

    def relu(self, a):
        idx = a <= 0
        a[idx] = 0.1 * a[idx]
        return a  

    def relu_deriv(self, a):
        a[a > 0] = 1.0
        a[a <= 0] = 0.1
        return a


# 神经网络层
class Layer:
    def __init__(self, num_neural, af_type="sigmoid", learn_rate=0.5):
        self.af_type = af_type  # active function type
        self.learn_rate = learn_rate
        self.num_neural = num_neural
        self.dim = 0
        self.W = np.zeros(5)

        self.a = None
        self.X = None
        self.z = None
        self.delta = None
        self.act_fun = ActivationFun(self.af_type)

    # 正向运算
    def fp(self, X):
        self.X = X
        xsize = X.shape
        self.dim = xsize[0]
        self.num = xsize[1]

        # 初始化
        if (self.W.all() == 0):
            if(self.af_type == "sigmoid"):
                self.W = np.random.normal(0, 1, size=(self.dim, self.num_neural)) / np.sqrt(self.num)
            elif(self.af_type == "relu"):
                self.W = np.random.normal(0, 1, size=(self.dim, self.num_neural)) * np.sqrt(2.0 / self.num)
        
        # 线性运算
        self.a = (self.W.T).dot(self.X)
        # 激活
        self.z = self.act_fun.cal(self.a)
        return self.z

    # bp运算
    def bp(self, delta):
        self.delta = delta * self.act_fun.cal_deriv(self.a)
        dW = self.X.dot(self.delta.T) / self.num
        self.W = self.W - self.learn_rate * dW
        delta_out = self.W.dot(self.delta)
        return delta_out

# 神经网络
class BpNet:
    def __init__(self, net_struct, stop_crit, max_iter, batch_size=10):
        self.name = "net work"
        self.net_struct = net_struct
        if len(self.net_struct) == 0:
            print("no layer is specified!")
            return

        self.stop_crit = stop_crit
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.layers = []
        self.num_layers = 0
        # 创建网络
        self.create_net(net_struct)
        self.loss_fun = LossFun("least_square")

    # 构建网络
    def create_net(self, net_struct):
        self.num_layers = len(net_struct)
        for i in range(self.num_layers):
            self.layers.append(Layer(net_struct[i][0], net_struct[i][1], net_struct[i][2]))

    # 训练
    def train(self, X, t, Xtest=None, ttest=None):
        eva_acc_list = []
        eva_loss_list = []

        xshape = X.shape
        num = xshape[0]
        dim = xshape[1]

        for k in range(self.max_iter):
            # 随机取出指定数目的数据
            idxs = random.sample(range(num), self.batch_size)
            xi = np.array([X[idxs, :]]).T[:, :, 0]
            ti = np.array([t[idxs, :]]).T[:, :, 0]

            zi = self.fp(xi)
            delta_i = self.loss_fun.cal_deriv(ti, zi)
            self.bp(delta_i)

            # 评估精度
            if (Xtest.any() != None):
                if k % 1000 == 0:
                    [eva_acc, eva_loss] = self.test(Xtest, ttest)
                    eva_acc_list.append(eva_acc)
                    eva_loss_list.append(eva_loss)
                    print ("%4d,%4f,%4f" % (k, eva_acc, eva_loss))
            else:
                print ("%4d" % (k))
        return [eva_acc_list, eva_loss_list]

    # 测试模型精度
    def test(self, X, t):
        xshape = X.shape
        num = xshape[0]
        z = self.fp(X.T)
        t = t.T
        est_pos = np.argmax(z, 0)
        real_pos = np.argmax(t, 0)
        corrct_count = np.sum(est_pos == real_pos)
        acc = 1.0 * corrct_count / num
        loss = self.loss_fun.cal(t, z)
        return [acc, loss]

    def fp(self, X):
        z = X
        for i in range(self.num_layers):
            z = self.layers[i].fp(z)
        return z

    def bp(self, delta):
        z = delta
        for i in range(self.num_layers - 1, -1, -1):
            z = self.layers[i].bp(z)
        return z

# 0中心化及标准化
def z_score_normalization(x):
    mu = np.mean(x)
    sigma = np.std(x)
    x = (x - mu) / sigma
    return x

def sigmoid(X, useStatus):
    if useStatus:
        return 1.0 / (1 + np.exp(-float(X)))
    else:
        return float(X)

def plot_curve(data, title, lege, xlabel, ylabel):
    num = len(data)
    idx = range(num)
    plt.plot(idx, data, color="r", linewidth=1)

    plt.xlabel(xlabel, fontsize="xx-large")
    plt.ylabel(ylabel, fontsize="xx-large")
    plt.title(title, fontsize="xx-large")
    plt.legend([lege], fontsize="xx-large", loc='upper left')
    plt.show()

if __name__ == "__main__":
    
    train_images = mnist.train_images()
    train_labels = mnist.train_labels()

    test_images = mnist.test_images()
    test_labels= mnist.test_labels()

    n_train, w, h = train_images.shape
    train_features = train_images.reshape( (n_train, w*h) )

    n_test, w, h = test_images.shape
    test_features = test_images.reshape( (n_test, w*h) )
    print("Import data OK")

    train_features = z_score_normalization(train_features)
    test_features = z_score_normalization(test_features)
    sample_num = train_labels.shape[0]
    tr_labels = np.zeros([sample_num, 10])
    for i in range(sample_num):
        tr_labels[i][train_labels[i]] = 1

    sample_num = test_labels.shape[0]
    te_labels = np.zeros([sample_num, 10])
    for i in range(sample_num):
        te_labels[i][test_labels[i]] = 1

    print("Data preprocessing OK")

    # Design the neural network
    stop_crit = 1000  # 停止次数
    batch_size = 100  # 每次训练的样本个数
    max_iter = batch_size * stop_crit  # 最大迭代次数
    net_struct = [[200,"sigmoid",0.05],[100,"relu",0.01],[10,"sigmoid",0.5]] # 网络结构[[batch_size,active function, learning rate]]

    # train
    bpNNCls = BpNet(net_struct, stop_crit, max_iter, batch_size)

    [acc, loss] = bpNNCls.train(train_features, tr_labels, test_features, te_labels)
    print("training model finished")

    # create test data
    plot_curve(acc, "Bp Network Accuracy", "accuracy", "iter", "Accuracy")
    plot_curve(loss, "Bp Network Loss", "loss", "iter", "Loss")


    # test
    [acc, loss] = bpNNCls.test(test_features, te_labels)
    print ("test accuracy:%.2f%" % (acc*100))