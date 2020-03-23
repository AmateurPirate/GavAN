from GavAN_Helpers import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.initializers import glorot_uniform
from keras.losses import binary_crossentropy

"""
  Level | Level for Humans | Level Description                  
 -------|------------------|------------------------------------ 
  0     | DEBUG            | [Default] Print all messages       
  1     | INFO             | Filter out INFO messages           
  2     | WARNING          | Filter out INFO & WARNING messages 
  3     | ERROR            | Filter out all messages 
"""

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
tf.logging.set_verbosity(tf.logging.ERROR)

class GavAN():
    def __init__(self):
        # Input shape
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        discriminator_optimizer = Adam(0.0002, 0.5)
        discriminator_loss = 'binary_crossentropy'
        discriminator_loss2 = tf.keras.losses.BinaryCrossentropy(label_smoothing=1)
        metrics = ['accuracy']

        combined_optimizer = Adam(0.0002, 0.5)
        combined_loss = 'binary_crossentropy'

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss=discriminator_loss2,
            optimizer=discriminator_optimizer,
            metrics=metrics)

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model(z, valid)
        self.combined.compile(loss=combined_loss, optimizer=combined_optimizer)

        # make output directory and save directory name as string
        self.output_directory = make_output_dir2()

    def build_generator(self):

        model = Sequential()

        model.add(Dense(1024, input_shape=(10 * 10,)))  # 1024,100
        model.add(Activation('tanh'))

        model.add(Dense(128 * 16 * 16))  # 128
        model.add(BatchNormalization())
        model.add(Activation('tanh'))

        model.add(Reshape((16, 16, 128)))

        model.add(Conv2DTranspose(64, (5, 5), activation='tanh', strides=2, padding='same'))
        # model.add(Conv2DTranspose(32, (5, 5), activation='tanh', strides=2, padding='same')) # add this in to make output (128, 128, 3)
        model.add(Conv2DTranspose(self.channels, (5, 5), activation='tanh', strides=2, padding='same'))

        print('----------------------------GENERATOR----------------------------')
        model.summary()
        print('\n')

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_generator(self):
        model = Sequential()

        model.add(Dense(1024, input_shape=(10 * 10,)))  # 1024,100
        model.add(Activation('tanh'))

        model.add(Dense(128 * 16 * 16))  # 128
        model.add(BatchNormalization())
        model.add(Activation('tanh'))

        model.add(Reshape((16, 16, 128)))

        model.add(Conv2DTranspose(64, (5, 5), activation='tanh', strides=2, padding='same'))
        model.add(Conv2DTranspose(32, (5, 5), activation='tanh', strides=2, padding='same')) # add this in to make output (128, 128, 3)
        model.add(Conv2DTranspose(self.channels, (5, 5), activation='tanh', strides=1, padding='same'))

        print('----------------------------GENERATOR----------------------------')
        model.summary()
        print('\n')

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):

        model = Sequential()

        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        print('--------------------------DISCRIMINATOR--------------------------')
        model.summary()
        print('\n')

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        X_train = load_data2('/Users/chenzhe/PycharmProjects/DCGAN1/formatted_arrays/ar6.npz')
        X_train = tf.squeeze(X_train)

        # Normalize between -1 and 1
        X_train = tf.cast(X_train, dtype=tf.float32)
        X_train = X_train / 127.5 - 1

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            imgs = batch_generator(X_train, batch_size=batch_size)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # print(("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss)))
            print("Epoch: {} [Discriminator loss: {}, Accuracy: {}%] [Generator loss: {}]".format(
                epoch, str(round(d_loss[0], 2)), str(int(round(100*d_loss[1]))), str(round(g_loss, 2))))


            # If at save interval => save generated image samples
            if epoch % save_interval == 0 and epoch != 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 4, 4
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = (0.5 * gen_imgs + 0.5)

        fig, axs = plt.subplots(r, c, figsize=(15, 15))

        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0:3])
                axs[i,j].axis('off')
                cnt += 1

        plt.tight_layout()
        plt.show()
        fig.savefig(self.output_directory + '/epoch_no_{}.png'.format(epoch))
        plt.close()

if __name__ == '__main__':
    gavan = GavAN()
    gavan.train(epochs=6, batch_size=32, save_interval=5)
    # gavan.train(epochs=4000, batch_size=32, save_interval=50) # defaults from tf model
