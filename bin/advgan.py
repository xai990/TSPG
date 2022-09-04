from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
import sklearn.datasets
import matplotlib.pyplot as plt

import sys

import numpy as np
import os
import umap.plot
import seaborn as sns
import cycler
import matplotlib.cm as cm
import argparse
import utils
from tensorflow import keras
import target_model


class GAN():
    def __init__(self,
                n_inputs,
                n_classes,
                classes,
                target,
                preload=False,
                output_dir = '.'):
        # save attributes
        self.n_inputs = n_inputs
        self.channels = 1
        self.gene_shape = (self.n_inputs, self.channels)
        self.target = target
        self.output_dir = output_dir
        self.classes = classes
        # load pre-trained target model
        self.target_model = target_model.load(output_dir)



        optimizer = Adam(0.0002, 0.5)
        # load models if specified
        if preload:
            self.load()

        else:
            # Build and compile the discriminator
            self.discriminator = self.build_discriminator()
            self.discriminator.compile(loss='mse',
                optimizer=optimizer,
                metrics=['accuracy'])

            # Build the generator
            self.generator = self.build_generator()

            # The generator takes noise as input and generates genes
            z = Input(shape=(self.n_inputs,))
            gene = self.generator(z)

            # For the combined model we will only train the generator
            self.discriminator.trainable = False

            # The discriminator takes generated images as input and determines validity
            validity = self.discriminator(gene)

            # The combined model  (stacked generator and discriminator)
            # Trains the generator to fool the discriminator
            self.combined = Model(z, validity)
            self.combined.compile(loss='mse', optimizer=optimizer)


    def build_generator(self):

        model = Sequential()

        model.add(Dense(256, input_dim=self.n_inputs))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.gene_shape), activation='tanh'))
        model.add(Reshape(self.gene_shape))

        model.summary()

        noise = Input(shape=(self.n_inputs,))
        gene = model(noise)

        return Model(noise, gene)

    def build_discriminator(self):

        model = Sequential()

        model.add(Flatten(input_shape=self.gene_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1))
        model.summary()

        gene = Input(shape=self.gene_shape)
        validity = model(gene)

        return Model(gene, validity)

    def train(self, X_train,y, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        #scaler = sklearn.preprocessing.MaxAbsScaler()
        #scaler.fit(X_train)

        #X_train = scaler.transform(X_train)
        X_train = np.expand_dims(X_train, axis=2)
        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        X_input = X_train[y != self.target]
        X_real = X_train[y == self.target]
        y = y.reshape(-1, 1)
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_real.shape[0], batch_size)
            genes = X_real[idx]

            noise = X_input[idx]

            # Generate a batch of new images
            gen_genes = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(genes, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_genes, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            #noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch, X_train,y, X_input, X_real)

    def sample_images(self, epoch, X_train,y, X_input, X_real):
        r, c = 10, 10
        #noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        idx = np.random.randint(0, X_input.shape[0], 32)
        noise = X_input[idx]
        x_fake = self.generator.predict(noise)
        #y = np.argmax(y, axis=1)
        reducer = umap.UMAP()
        x_merged =np.vstack( [X_train,x_fake])
        x_merged = np.squeeze(x_merged, axis=2)
        embedding = reducer.fit_transform(x_merged)

        x      = embedding[:len(X_train)]
        #print(x.shape)
        x_pert = embedding[len(X_train):]
        classes = self.classes
        os.makedirs("images", exist_ok=True)

        plt.axis('off')
        plt.gca().set_prop_cycle(cycler.cycler(color=cm.get_cmap('tab10').colors))
        for k in range(len(classes)):
            indices = (y == k)
            #print(indices.shape)
            if np.any(indices):
                plt.scatter(
                    x[indices[:,0], 0],
                    x[indices[:,0], 1],
                    label=classes[k],
                    edgecolors='w')

        plt.gca().set_prop_cycle(cycler.cycler(color=cm.get_cmap('tab10').colors))
        plt.scatter(
                    x_pert[:, 0],
                    x_pert[:, 1],
                    alpha=0.25,
                    label='%s (perturbed)' % (classes[k]))

        plt.tight_layout()
        plt.subplots_adjust(right=0.70)
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig("images/%d.png" % epoch)
        plt.close()


    def perturb(self, x, y):
        #screen the samples
        y_c = np.argmax(y, axis=1)
        x_p = x [np.argmax(y, axis=1) != self.target]
        # compute perturbations, perturbed samples
        x_fake = self.generator.predict(x_p)
        x_fake = np.squeeze(x_fake, axis=2)
        p  = []
        x_new = []
        i = 0
        for _ in range(len(y_c)):
            # samples are not target
            if (y_c[_] != self.target):
                p.append(x_fake[i] - x[_])
                x_new.append(x_fake[i])
                i+=1
            # samples are target so the p values are 0
            else:
                p.append(x[_] - x[_])
                x_new.append(x[_])


        x_new = np.array(x_new)
        p = np.array(p)

        return x_new, p



    def predict_target(self, x):
        return self.target_model(x, training=False)




    def score(self, x, y):

        # compute perturbed samples
        x_fake, p = self.perturb(x,y)

        # feed perturbed samples to target model
        y_fake = self.predict_target(x_fake)

        # compute perturbation accuracy
        score = np.mean(np.argmax(y_fake, axis=1) == self.target)

        return score, x_fake, p, y_fake



    def save(self):
        # initialize output directory
        os.makedirs(self.output_dir, exist_ok=True)

        # save generator, discriminator
        self.generator.save('%s/%d_generator.h5' % (self.output_dir, self.target))
        self.discriminator.save('%s/%d_discriminator.h5' % (self.output_dir, self.target))

    def load(self):
        self.generator = keras.models.load_model('%s/%d_generator.h5' % (self.output_dir, self.target), compile=False)
        self.discriminator = keras.models.load_model('%s/%d_discriminator.h5' % (self.output_dir, self.target), compile=False)
