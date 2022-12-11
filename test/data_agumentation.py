from cifar10 import *
from PIL import Image
import matplotlib.pyplot as plt
import os

(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=True, flatten=False, one_hot_label=True)
print(t_train.shape)

t_train = np.append(t_train,t_train,axis=0)
print(t_train.shape)

def img_augment(x_train, t_train):

    dim = x_train.shape
    train_set = x_train
    print(dim[0])

    #Rotate 90
    for i in range(dim[0]):
        print(i)
        img = x_train[0] * 255
        img = np.transpose(img, (1,2,0))
        img = Image.fromarray(img.astype(np.uint8))
        img = img.rotate(90)

        img = np.array(img,dtype=np.float32)
        img = img/255.0
        img = np.transpose(img, (2,0,1))
        img = img[np.newaxis,:]
        np.append(train_set,img,axis=0)
        os.system("cls")
    
    return train_set


t_set = img_augment(x_train,t_train)
img = t_set[500010] * 255
img = np.transpose(img,(1,2,0))
img = Image.fromarray(img.astype(np.uint8))
img.show()



# img = x_train[0] * 255
# img = np.transpose(img, (1,2,0))
# img = Image.fromarray(img.astype(np.uint8))
# img = img.rotate(90)
# img.show()

# img = np.array(img,dtype=np.float32)
# img = img/255.0
# img = np.transpose(img, (2,0,1))
# print(img.shape)
