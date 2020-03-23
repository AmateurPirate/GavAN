from PIL import Image
import os
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from GavAN_Helpers import load_data2
from keras.datasets import mnist
import random

tf.compat.v1.enable_eager_execution()

path1 = '/Users/chenzhe/Desktop/passportphoto.jpeg'
path2 = '/Users/chenzhe/Desktop/spag.jpeg'
path3 = '/Users/chenzhe/Desktop/psy3.png'
path4 = '/Users/chenzhe/Desktop/duff4.jpeg'

def effect1(image_path, skew=1, bias=0.01, save=False, subtraction=False, addition=False, multiply=False, multiply2=False):
    im = np.asarray(Image.open(image_path))
    if im.ndim == 2:
        im = np.expand_dims(im, axis=2)
    if subtraction:
        x_var = im[:, skew:, :] - im[:, :-skew, :]
        y_var = im[skew:, :, :] - im[:-skew, :, :]
    if addition:
        x_var = (im[:, skew:, :] + im[:, :-skew, :]) / 2
        y_var = (im[skew:, :, :] + im[:-skew, :, :]) / 2
    if multiply:
        x_var = (im[:, skew:, :] * im[:, :-skew, :]) / (im[:, skew:, :] + bias)
        y_var = (im[skew:, :, :] * im[:-skew, :, :]) / (im[skew:, :, :] + bias)
    if multiply2:
        x_var = (im[skew:, skew:, :] * im[skew:, :-skew, :]) / (im[skew:, skew:, :] + bias)
        y_var = (im[skew:, skew:, :] * im[:-skew, skew:, :]) / (im[skew:, skew:, :] + bias)
    # x_var = x_var.astype(np.float32)
    # y_var = y_var.astype(np.float32)
    im_x = Image.fromarray(x_var, mode='RGB')
    im_y = Image.fromarray(y_var, mode='RGB')
    if save:
        im_num = random.randint(1000, 9999)
        im_x.save(os.getcwd() + '/' + str(im_num) + 'x' + '.png')
        im_y.save(os.getcwd() + '/' + str(im_num) + 'y' + '.png')
    im_x.show()
    im_y.show()
    return x_var, y_var

arr_x, arr_y = effect1(path2, skew=1, bias=0.01, subtraction=True, addition=False, multiply=False, multiply2=False, save=False)

print(arr_x.shape)

def kernel_interpolation(arr, kernel_shape='x', save=False): # kernel_shape can be 'x' or 'cross'
    for r in range(1, arr.shape[0]-1):
        for c in range(1, arr.shape[1]-1):
            for chan in range(arr.shape[2]):
                if arr[r, c, chan] == 0 or arr[r, c, chan] == 255:
                    if kernel_shape == 'x':
                        arr[r, c, chan] = sum([arr[r-1, c-1, chan], arr[r+1, c-1, chan], arr[r-1, c+1, chan], arr[r+1, c+1, chan]]) / 4
                    elif kernel_shape == 'cross':
                        arr[r, c, chan] = sum([arr[r-1, c, chan], arr[r, c-1, chan], arr[r, c+1, chan], arr[r+1, c, chan]]) / 4
    im = Image.fromarray(arr, mode='RGB')
    if save:
        im_num = random.randint(1000, 9999)
        im.save(os.getcwd() + '/' + str(im_num) + 'KI' + '.png')
    im.show()
    return

kernel_interpolation(arr_y, kernel_shape='cross')
