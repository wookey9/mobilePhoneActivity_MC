from keras.models import Sequential
from keras.layers import Conv3D
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.callbacks import ModelCheckpoint
from keras.models import model_from_json
import tensorflow.compat.v1 as tf
import gc
import time

from keras.applications.vgg16 import VGG16
from keras.models import Model

import sys
import numpy as np
import PIL.Image as pilimg
import pylab as plt

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

fileaddress = "./image_sliding/MILAN_UP_RATIO_" # 이미지 위치

thenumberoffile = 147 # the number of images

time_step = 5 # 타임스탭



row = 12 # 이미지의 사이즈 CAIDA 데이터셋일 경우 row = 15, col = 20
col = 12

epoch = 100 # 이포크

seq = Sequential()
seq.add(ConvLSTM2D(filters=40, kernel_size=(2, 2),
                   input_shape=(time_step, row, col, 1),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(2, 2),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(2, 2),
                   padding='same', return_sequences=True))
seq.add(BatchNormalization())

seq.add(ConvLSTM2D(filters=40, kernel_size=(2, 2),
                   padding='same'))
seq.add(BatchNormalization())

seq.add(Conv2D(filters=1, kernel_size=(3, 3),
               activation='sigmoid',
               padding='same', data_format='channels_last'))
seq.compile(loss='mse', optimizer='adamax')


def generate_movies6(filenum, n_frames): # 학습을 위해 이미지 -> 행렬로 바꾸는 과정 위 과정 이후 학습을 시작함
    current_movies = np.zeros((filenum - n_frames, n_frames, row, col, 1), dtype=float)
    predict_movies = np.zeros((filenum - n_frames, row, col, 1), dtype=float)
    for j in range(filenum - n_frames):
        for f in range(n_frames):
            if j + f < filenum + 1:
                current_image = pilimg.open(fileaddress + str(j + f) + '.png')
                pix = np.array(current_image)
                current_movies[j, f, :, :, 0] = pix.copy()
                current_movies[j, f, :, :, 0] = current_movies[j, f, :, :, 0] / 255
                print(current_movies[j, f, :, :, 0])
                if (f == n_frames - 1):
                    predict_image = pilimg.open(
                        fileaddress + str(j + f + 1) + '.png')
                    pix = np.array(predict_image)
                    predict_movies[j, :, :, 0] = pix.copy()
                    predict_movies[j, :, :, 0] = predict_movies[j, :, :, 0] / 255
                    print(predict_movies[j, :, :, 0])
                print(j + f)
    return current_movies, predict_movies # Current는 입력이미지들(timesteps) Predict는 라벨

def prediction_part9():
    arr_RMSE = []
    for which in range(x_test.shape[0]):
        track = x_test[which][::, ::, ::, ::]
        start1 = time.time()
        new_pos = seq.predict(track[np.newaxis, ::, ::, ::, ::])
        print("time :", time.time() - start1)

        RMSE = mean_squared_error(y_test[which][6:, ::, 0], new_pos[0, 6:, ::, 0]) # 뒤에서 8분에 대한 RMSE 만약 12분이라면 13->12가 되어야하고 16분이면 13->11이 되어야함
        arr_RMSE.append(RMSE)

        fig = plt.figure(figsize=(5, 5))
        toplot3 = y_test[which][::, ::, 0]
        plt.imshow(toplot3, cmap='gray', vmin=0, vmax=1)
        plt.savefig('./results/%i_ground_truth.png' % (which))
        toplot = new_pos[0, ::, ::, 0]
        plt.imshow(toplot, cmap='gray', vmin=0, vmax=1)
        plt.savefig('./results/%i_predicted.png' % (which))
        diff_fig = new_pos[0, ::, ::, 0] - y_test[which][::, ::, 0]
        abs_diff_fig = abs(diff_fig)
        abs_diff_fig = 1 - abs_diff_fig
        toplot2 = abs_diff_fig
        plt.imshow(toplot2, cmap='gray', vmin=0, vmax=1)
        plt.savefig('./results/diffimg_%i.png' % (which))
        temp = new_pos[0, ::, ::, 0]
        temp[:, :] = 1 - temp[:, :]
        temp2 = y_test[which][::,::,0]
        temp2[:,:] =  1 - temp2[:,:]
        np.savetxt('./results/%i_ground_truth.txt' % (which), temp2, fmt="%1.5f")
        np.savetxt('./results/%i_predicted.txt' % (which), temp, fmt="%1.5f")
    np.savetxt('./results/%RMSE.txt' , arr_RMSE, fmt='%1.9f')


current_movies, predict_movies = generate_movies6(filenum=thenumberoffile, n_frames=time_step)

x_train, x_test, y_train, y_test = train_test_split(current_movies, predict_movies, test_size=0.2, shuffle=True) ## 트레이닝시 셔플하지않으면 학습이 잘안됨

start = time.time()
hist = seq.fit(x_train, y_train, batch_size=10, epochs=epoch, validation_split=0.1, verbose=2, shuffle=False) ## 데이터셋을 셔플로 나누면 학습이 잘됨 하지만 고정된 테스트 데이터셋을 얻지못함
print("time :", time.time() - start)

# seq.save_weights("./results/weights/model_%i.h5" %(time_step)) // 가중치 저장

# seq.load_weights("./results/weights/CAIDA/disjoint/model_%i.h5" %(time_step)) // 가중치 불러오기

y_vloss = hist.history['val_loss']
y_loss = hist.history['loss']
x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")
plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
# plt.show()
plt.savefig('./results/myplot.png')

prediction_part9()

