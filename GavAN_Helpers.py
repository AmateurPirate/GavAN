import datetime
import os
from PIL import Image
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import PIL.Image as Image

# tf.compat.v1.enable_eager_execution()

dir_path = '/Users/chenzhe/PycharmProjects/DCGAN1/rare_pepes_128/'

def auto_resize(from_dir_path, to_dir_path, image_height=128, image_width=128):
    Path(to_dir_path).mkdir(parents=True, exist_ok=True)
    count = 0
    for image_file_name in os.listdir(from_dir_path):
        count += 1
        if image_file_name.endswith('.jpg') or image_file_name.endswith('.png') or image_file_name.endswith('.jpeg'):
            im = Image.open(from_dir_path + image_file_name).convert('RGB')
            new_width = 128
            new_height = 128
            im = im.resize((new_height, new_width), Image.ANTIALIAS)
            im.save(to_dir_path + str(count) + '.jpg')

def batch_generator(imgs_tensor, batch_size=32):
    idx = np.random.randint(0, imgs_tensor.shape[0], batch_size)
    c = tf.stack([imgs_tensor[idx[0], :, :, :], imgs_tensor[idx[1], :, :, :]])
    for i in range(2, batch_size):
        a = c
        b = tf.expand_dims(imgs_tensor[idx[i], :, :, :], axis=0)
        c = tf.concat([a, b], axis=0)
    return c

def crop(path, input, height, width):
    im = Image.open(input)
    imgwidth, imgheight = im.size
    k = 1
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            a.save(os.path.join(path, 'rare_pepes_2/', "IMG-%s.png" % k))
            k +=1

def high_pass_x_y(image):
    x_var = image[:,:,1:,:] - image[:,:,:-1,:]
    y_var = image[:,1:,:,:] - image[:,:-1,:,:]
    return x_var, y_var

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)

    plt.imshow(image)
    if title:
        plt.title(title)
    plt.show()

def load_data(file_path):
    data_array = np.load('ar3.npz')['arr_0'].astype(np.uint8, casting='unsafe')
    data_array = data_array / 255 # Normalize between 0 and 1
    data_array = np.expand_dims(data_array, axis=4) # Add batch dimension
    return data_array

def load_data2(file_path):
    data_array = np.load(file_path)['arr_0'].astype(np.uint8, casting='unsafe')
    # data_array = data_array / 127.5 - 1
    data_array = np.expand_dims(data_array, axis=4) # Add batch dimension
    return data_array

def make_input_array(dir_path, image_height=128, image_width=128, channels=3): # to load file use np.load('filename.npz')['arr_0']
    M = np.zeros(shape=[1, image_height, image_width, channels])
    for image_file_name in os.listdir(dir_path):
        if image_file_name.endswith('jpg'):
            img = tf.io.read_file(dir_path + image_file_name)
            img = tf.image.decode_jpeg(img, channels=channels)
            img = tf.image.resize(img, [image_height, image_width])
            img = tf.cast(img, tf.float32)
            # img = img / 0.5 - 1
            # img = img / 127.5 - 1
            # img = img / 255 - 1
            img = tf.expand_dims(img, axis=0)
            img = img.numpy()
            M = np.concatenate((M, img), axis=0)
    return M[1:, :, :, :].astype('uint8')

def make_noise(image):
    return tf.random.normal(shape=tf.shape(image))

def make_output_dir(model_name='model_output', timestamp=False):
    if timestamp:
        now = datetime.datetime.ctime(datetime.datetime.now())
        timestamp = (now[4:16] + '_' + now[-4:]).replace(' ', '_').replace(':', '_')
        Path(str(model_name) + '_' + str(timestamp)).mkdir(parents=True, exist_ok=True)
    else:
        Path(str(model_name)).mkdir(parents=True, exist_ok=True)

def make_output_dir2():
    if 'model_output' not in os.listdir(os.getcwd()):
        Path(os.getcwd() + '/model_output/try_0').mkdir(parents=True, exist_ok=True)
        return os.getcwd() + '/model_output/try_0'
    else:
        highest = 0
        for filename in os.listdir(os.getcwd() + '/model_output'):
            if filename != '.DS_Store':
                highest = max(highest, int(filename[4]))
        Path(os.getcwd() + '/model_output/try_{}'.format(str(highest + 1))).mkdir(parents=True, exist_ok=True)
        return os.getcwd() + '/model_output/try_{}'.format(str(highest + 1))

def print_images(array, num_of_images=1):
    for i in range(num_of_images):
        image1 = array[i, :, :, :]
        tensor_to_image(image1).show()

def save_array_as_npz(array):
    np.savez_compressed('ar3.npz', array)

def tensor_to_image(tensor):
    # tensor = (tensor + 1) * 127.5
    # tensor = (tensor + 1) * 255
    tensor = np.array(tensor, dtype=np.uint8)
    tensor = np.squeeze(tensor)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)

# ar = make_input_array(dir_path)
# save_array_as_npz(ar)

