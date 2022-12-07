# coding: utf-8
#!/usr/bin/python3

import sys, os

#파이썬 실행파일 위치로 이동
dataset_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(dataset_dir)

#부모파일 import
sys.path.append(os.pardir)
import pickle
import numpy as np

from collections import OrderedDict

eps = 1e-17         #0에 수렴하는 작은 값 설정(NoneType Error 대비)

def softmax(x):     #소프트맥스 함수
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))

def cross_entropy_error(y, t):  
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)

    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + eps)) / batch_size

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).
    
    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """(im2col과 반대) 2차원 배열을 입력받아 다수의 이미지 묶음으로 변환한다.
    
    Parameters
    ----------
    col : 2차원 배열(입력 데이터)
    input_shape : 원래 이미지 데이터의 형상（예：(10, 1, 28, 28)）
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩
    
    Returns
    -------
    img : 변환된 이미지들
    """
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)

    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]



class ELU:              #은닉층 활성함수
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        self.out = x
        self.out[self.out <= 0] = np.exp(self.out[self.out <= 0]) - 1
        return self.out

    def backward(self,dout):
        self.out[self.out >0] = 1
        self.out[self.out <= 0] += 1
        return self.out * dout

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
            
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        
        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        
        return dx

class Adam:             #파라메터 옵티마이저
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)
        
        self.iter += 1
        lr_t  = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)         
        
        for key in params.keys():
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key]**2 - self.v[key])
            
            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + eps)

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None

    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x

        out = np.dot(self.x, self.W) + self.b

        return out

    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

class Convolution:
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
        
        # 중간 데이터（backward 시 사용）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 가중치와 편향 매개변수의 기울기
        self.dW = None
        self.db = None

    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = 1 + int((H + 2*self.pad - FH) / self.stride)
        out_w = 1 + int((W + 2*self.pad - FW) / self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T

        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        self.x = x
        self.col = col
        self.col_W = col_W

        return out

    def backward(self, dout):
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
        
        self.x = None
        self.arg_max = None

    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 + (H - self.pool_h) / self.stride)
        out_w = int(1 + (W - self.pool_w) / self.stride)

        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)
        col = col.reshape(-1, self.pool_h*self.pool_w)

        arg_max = np.argmax(col, axis=1)
        out = np.max(col, axis=1)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)

        self.x = x
        self.arg_max = arg_max

        return out

    def backward(self, dout):
        dout = dout.transpose(0, 2, 3, 1)
        
        pool_size = self.pool_h * self.pool_w
        dmax = np.zeros((dout.size, pool_size))
        dmax[np.arange(self.arg_max.size), self.arg_max.flatten()] = dout.flatten()
        dmax = dmax.reshape(dout.shape + (pool_size,)) 
        
        dcol = dmax.reshape(dmax.shape[0] * dmax.shape[1] * dmax.shape[2], -1)
        dx = col2im(dcol, self.x.shape, self.pool_h, self.pool_w, self.stride, self.pad)
        
        return dx

class LeNet5:
    """
    단순한 합성곱 신경망
    conv - relu - pool - affine - relu - affine - softmax
    """

    def __init__(self, input_dim=(1, 28, 28),
                 conv_param_1={'filter_num': 10, 'filter_size': 5, 'pad': 2, 'stride': 1},
                 conv_param_2={'filter_num': 50, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 conv_param_3={'filter_num': 150, 'filter_size': 5, 'pad': 0, 'stride': 1},
                 weight_init_std=0.01):

        # 가중치 초기화 filter_num, input_dim[0], filter_size, filter_size
        self.params = {}
        self.params['W1'] = weight_init_std * \
                            np.random.randn(10, 3, 5, 5)
        self.params['b1'] = np.zeros(10)
        self.params['W3'] = weight_init_std * \
                            np.random.randn(50, 10, 5, 5)
        self.params['b3'] = np.zeros(50)
        self.params['W5'] = weight_init_std * \
                            np.random.randn(150, 50, 5, 5)
        self.params['b5'] = np.zeros(150)
        self.params['W6'] = weight_init_std * \
                            np.random.randn(150, 480)
        self.params['b6'] = np.zeros(480)
        self.params['W7'] = weight_init_std * \
                            np.random.randn(480, 200)
        self.params['b7'] = np.zeros(200)
        self.params['W8'] = weight_init_std * \
                            np.random.randn(200,100)
        self.params['b8'] = np.zeros(100)
        self.params['W9'] = weight_init_std * \
                            np.random.randn(100,50)
        self.params['b9'] = np.zeros(50)
        self.params['W10'] = weight_init_std * \
                            np.random.randn(50,25)
        self.params['b10'] = np.zeros(25)
        self.params['W11'] = weight_init_std * \
                            np.random.randn(25,10)
        self.params['b11'] = np.zeros(10)
        



        # 계층 생성
        self.layers = OrderedDict()
        # C1 : 컨볼루션 연산
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param_1['stride'], conv_param_1['pad'])
        self.layers['ReLU_1'] = ELU()
        # S2 : 풀링 계층 (LeNet에서는 평균풀링을 사용했으나, 여기서는 최대풀링)
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # C3 : 컨볼루션 연산
        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv_param_2['stride'], conv_param_2['pad'])
        self.layers['ReLU_2'] = ELU()
        # S4 : 풀링 계층
        self.layers['Pool4'] = Pooling(pool_h=2, pool_w=2, stride=2)
        # C5 : 컨볼루션 연산
        self.layers['Conv5'] = Convolution(self.params['W5'], self.params['b5'],
                                           conv_param_3['stride'], conv_param_3['pad'])

        self.layers['ReLU_3'] = ELU()
        # F6 : Affine 계층 120 -> 84
        self.layers['Affine6'] = Affine(self.params['W6'], self.params['b6'])

        self.layers['ReLU_4'] = ELU()
        # F7 : Affine 계층 84 -> 200
        self.layers['Affine7'] = Affine(self.params['W7'], self.params['b7'])

        self.layers['ReLU_5'] = ELU()
        # F8 : Affine 계층 200 -> 100
        self.layers['Affine8'] = Affine(self.params['W8'], self.params['b8'])

        self.layers['ReLU_6'] = ELU()
        # F9 : Affine 계층 100 -> 50
        self.layers['Affine9'] = Affine(self.params['W9'], self.params['b9'])

        self.layers['ReLU_7'] = ELU()
        # F9 : Affine 계층 50 -> 25
        self.layers['Affine10'] = Affine(self.params['W10'], self.params['b10'])

        self.layers['ReLU_8'] = ELU()
        # F9 : Affine 계층 25 -> 10
        self.layers['Affine11'] = Affine(self.params['W11'], self.params['b11'])
        self.last_layer = SoftmaxWithLoss()


    def predict(self, x):
        for layer in self.layers.values():
            # print(x.shape)
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1: t = np.argmax(t, axis=1)

        acc = 0.0

        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i * batch_size:(i + 1) * batch_size]
            tt = t[i * batch_size:(i + 1) * batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y == tt)

        return acc / x.shape[0]

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W5'], grads['b5'] = self.layers['Conv5'].dW, self.layers['Conv5'].db
        grads['W6'], grads['b6'] = self.layers['Affine6'].dW, self.layers['Affine6'].db
        grads['W7'], grads['b7'] = self.layers['Affine7'].dW, self.layers['Affine7'].db
        grads['W8'], grads['b8'] = self.layers['Affine8'].dW, self.layers['Affine8'].db
        grads['W9'], grads['b9'] = self.layers['Affine9'].dW, self.layers['Affine9'].db
        grads['W10'], grads['b10'] = self.layers['Affine10'].dW, self.layers['Affine10'].db
        grads['W11'], grads['b11'] = self.layers['Affine11'].dW, self.layers['Affine11'].db

        return grads

    def save_params(self, file_name="params_Lenet.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params_Lenet.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

            j = 1

        for i, key in enumerate(['Conv1', 'Conv3', 'Conv5', 'Affine6', 'Affine7', 'Affine8', 'Affine9', 'Affine10', 'Affine11']):
            self.layers[key].W = self.params['W' + str(i + j)]
            self.layers[key].b = self.params['b' + str(i + j)]
            if j < 3:
                j += 1