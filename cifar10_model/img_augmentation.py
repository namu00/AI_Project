from PIL import Image
from dataset.cifar10 import *
import numpy as np
import os, platform

def img_augment(x_train, t_train, batch_size):
    train_set = x_train
    for i in range(batch_size):
        img = x_train[i] * 255                               #Normalize 해제
        img = np.transpose(img, (1, 2, 0))                   #차원 변환, C/H/W --> H/W/C
        img = Image.fromarray(img.astype(np.uint8))          #이미지 저장을 위해 넘파이 배열타입을 정수형으로 변환
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT) #이미지 좌/우 반전
        #좌/우 반전만 Augmenting하는 이유:
        #90/180/270도 회전하여 학습한 경, 레퍼런스 입력모델 대비 출력값 향상이 아~~~~주 미미함

        img = np.array(img, dtype=np.float32)           #Numpy배열로 변경하기 위한 형변환, uint8 --> float32
        img = img/255.0                                 #Normalize 실행
        img = np.transpose(img, (2, 0, 1))              #차원 변환, H/W/C --> C/H/W
        img = img[np.newaxis, :]                        #기존 학습데이터셋에 추가하기 위한 빈 N채널 생성, C/H/W --> N/C/H/W
        train_set = np.append(train_set, img, axis=0)   #기존 학습데이터셋에 Augmenting 데이터 추가
    
    label_set = np.append(t_train, t_train, axis=0)     #label값 확장(똑같은 사진정보를 두 번 갖다붙힌 작업이라서, t_train을 두 번 붙힘)
    return (train_set, label_set)


def convert_img(np_array,normalize=True):
    img = np_array
    if normalize:
        img = img*255
    
    img = np.transpose(img,(1,2,0))
    img = Image.fromarray(img.astype(np.uint8))
    return img

if __name__ == "__main__": #함수 테스트
    (x_train, t_train), (x_test, t_test) = load_cifar10(
        normalize=True, flatten=False, one_hot_label=True)

    test_range = x_train.shape[0]
    t_set, t_label = img_augment(x_train, t_train,test_range)

    # img1 = convert_img(t_set[0])
    # img1.show()
    # img2 = convert_img(t_set[50000])
    # img2.show()

    print(t_set.shape)
    print(t_label.shape)

    for i in range(test_range):
        print("Converting Image... Index: %5d"% (i))
        img1 = convert_img(t_set[i])
        img2 = convert_img(t_set[(i+test_range)])
        img2 = img2.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        label1 = np.argmax(t_label[i])
        label2 = np.argmax(t_label[(i + test_range)])

        if img1 != img2 :
            print("Image Dosen't match!")
            break

        elif label1 != label2 :
            print("Label dosen't match!")
            break

        if platform.system() == "Windows": os.system("cls")
        else: os.system("clear")

        if (i == test_range -1):
            print("No Issues on Image Augmenting")
            break