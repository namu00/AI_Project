# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from model import *
from dataset.fashion_mnist import *
from img_augmentation import *

# for reproducibility
np.random.seed(0)

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_fashionMNIST(normalize=True, flatten=False, one_hot_label=True)

pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)

network = LeNet5(input_dim=(1, 28, 28),
                  conv_param_1={'filter_num': 5, 'filter_size': 5, 'pad': 2, 'stride': 1},
                  conv_param_2={'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                  conv_param_3={'filter_num': 150, 'filter_size': 5, 'pad': 0, 'stride': 1},
                  weight_init_std=0.01)

# optimizer = SGD()
optimizer = Adam(lr=0.001)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

print("\tTrain Fashion_MNIST. . .")
for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    (x_batch, t_batch) = img_augment(x_batch,t_batch,batch_size)

    
    # 기울기 계산
    grads = network.gradient(x_batch, t_batch) # 오차역전파
    params = network.params

    # 갱신
    optimizer.update(params, grads)
    
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print('iter: (%5d:%5d) train acc: %3.4f test acc: %3.4f' % (i, iters_num, train_acc, test_acc))
        
        path_dir = './ckpt'
        file_name = "lenet5_fashion_params.pkl"
        if not os.path.isdir(path_dir):
            os.mkdir(path_dir)

        network.save_params(os.path.join(path_dir, file_name))

# 그래프 그리기
markers = {'train': 'o', 'test': 's'}
x = np.arange(len(train_acc_list))
plt.plot(x, train_acc_list, label='train acc')
plt.plot(x, test_acc_list, label='test acc', linestyle='--')
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()
