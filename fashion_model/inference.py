# coding: utf-8
import sys, os
sys.path.append(os.pardir)

import numpy as np
import matplotlib.pyplot as plt

from model import *
from dataset.fashion_mnist import *


# 데이터 읽기
os.chdir("./dataset")
(x_train, t_train), (x_test, t_test) = load_fashionMNIST(normalize=True, flatten=False, one_hot_label=True)
pwd = os.path.dirname(os.path.abspath(__file__))
os.chdir(pwd)

network = LeNet5(input_dim=(1, 28, 28),
                  conv_param_1={'filter_num': 10, 'filter_size': 5, 'pad': 2, 'stride': 1},
                  conv_param_2={'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                  conv_param_3={'filter_num': 150, 'filter_size': 5, 'pad': 0, 'stride': 1},
                  weight_init_std=0.01)

# 매개변수 가져오기
path_dir = './ckpt'
file_name = "lenet5_fashion_params.pkl"
network.load_params(os.path.join(path_dir, file_name))
print("Parameter Load Complete!")

test_acc = network.accuracy(x_test, t_test)
print("test acc | ", format(test_acc*100, ".4f"), '%')
