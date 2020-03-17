#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import datetime as dt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from tqdm import tqdm
from keras import backend as K
from itertools import product
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from keras.models import model_from_json
import tensorflow as tf

K.set_floatx('float64')

###############################
##### Import des donnees ######
###############################

data_close = pd.read_csv('data.csv')

n_time = data_close.shape[0]
n_stocks = data_close.shape[1]

## Inference par la moyenne par colomne
for j in range(0,n_stocks):
        data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())


## Creation du dataframe des rendements
rendements = np.log(data_close.values[1:,:]/data_close.values[:-1,:])
#rendements = np.transpose(rendements)

X_train_r, X_test_r = train_test_split(rendements, test_size = 0.33, random_state = 42)

X_size = X_train_r.shape[1]
noise_size = 1

#######################
#### GAN algorithm ####
#######################

## generator
def create_generator(nb_neurone):
    generator=Sequential()
    generator.add(Dense(units=nb_neurone[0],input_dim=noise_size,activation="relu"))
    #generator.add(ReLU(0.2))
    for i in nb_neurone[1:]:
        generator.add(Dense(units=i,activation="relu"))                                                                                        
    generator.add(Dense(units=X_size,activation="tanh"))   
    generator.summary()
    return generator

g=create_generator([1])


## discriminator
def create_discriminator(nb_neurone):
    discriminator=Sequential()
    discriminator.add(Dense(units=nb_neurone[0],input_dim=X_size,activation="relu"))
    discriminator.add(Dropout(0.3))
    for i in nb_neurone[1:]:
        discriminator.add(Dense(units=i,activation="relu"))
        discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

d = create_discriminator([1])

def create_encoder(nb_neurone):
    encoder=Sequential()
    encoder.add(Dense(units=nb_neurone[0],input_dim=X_size,activation="relu"))
    encoder.add(Dropout(0.3))
    for i in nb_neurone[1:]:
        encoder.add(Dense(units=int(i),activation="relu"))
        encoder.add(Dropout(0.3))
    encoder.add(Dense(units=noise_size, activation='linear'))
    encoder.summary()
    encoder.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return encoder

e = create_encoder([1])

def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(noise_size,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = create_gan(d,g)

## Score du KDE
def KDE(x,y):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x)
    score = sum(kde.score_samples(y[:,]))
    return score

KDE(X_train_r,X_test_r)


## Train our GAN model
def training(distrib='normal',epochs=500, batch_size=X_size,nb_neurone=20):
    # Creating GAN
    generator= create_generator([nb_neurone])
    discriminator= create_discriminator([nb_neurone])
    gan = create_gan(discriminator, generator)

    for e in range(1,epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            if distrib == 'normal':
                noisetensor = tf.random_normal(shape=[batch_size,noise_size],dtype=tf.float64)
            else:
                noisetensor = tf.random_uniform(shape=[batch_size,noise_size],dtype=tf.float64)
            with tf.Session() as sess:
                noise = sess.run(noisetensor)
            generated_indexes = generator.predict(noise)
            
            indextensor = tf.random.uniform([batch_size], 0 ,X_train_r.shape[0],dtype=tf.int64)
            with tf.Session() as sess:
                index = sess.run(indextensor)
            image_batch =X_train_r[index]
            
            concattensor = tf.concat([image_batch,generated_indexes],0)
            with tf.Session() as sess:
                X = sess.run(concattensor)
            
            y_distensor = tf.zeros([2*batch_size], tf.int64) 
            with tf.Session() as sess:
                y_dis = sess.run(y_distensor)
            y_dis[:batch_size]=1
            
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            
            if distrib == 'normal':
                noisetensor = tf.random_normal(shape=[batch_size,noise_size],dtype=tf.float64)
            else:
                noisetensor = tf.random_uniform(shape=[batch_size,noise_size],dtype=tf.float64)
            with tf.Session() as sess:
                noise = sess.run(noisetensor)
                
            y_gentensor = tf.ones([batch_size], tf.int64) 
            with tf.Session() as sess:
                y_gen = sess.run(y_gentensor)
            discriminator.trainable=False
            gan.train_on_batch(noise, y_gen)
    return gan, generator, discriminator

training(distrib='normal',epochs=1, batch_size=X_size,nb_neurone=20)

def training_encoder(generator, neurones,epochs=500):
    label = np.random.normal(0,1, [X_size,noise_size])
    X = generator.predict(label)
    label2 = np.random.normal(0,1, [X_size,noise_size])
    X2 = generator.predict(label2)
    encoder = create_encoder(neurones)
    # Fit the model
    encoder.fit(X, label, epochs=epochs)
    # evaluate the model
    scores = encoder.evaluate(X2, label2)
    print("\n%s: %.2f" % (encoder.metrics_names[0], scores[0]))
    return scores[1]

gan, generator, discriminator = training(epochs=500)

def find_optimal_model(C=3,N=4,epoch=1):
    #gan, generator, discriminator = training(epochs=epoch)
    neurones = [10*x for x in range(1,N+1)]
    scores = []
    
    for i in range(1,C+1):
        for neurons in product(neurones, repeat=i):
            nom = ""
            for k in neurons :
                nom = nom + " C"+str(k)
            tmp = [nom,training_encoder(generator, neurons,epochs=epoch)]
            scores.append(tmp)
    return scores

res=find_optimal_model(3,4,500)                                                                               
pd.DataFrame(res).to_csv('res.csv',index = False)

def train_encoder(generator, neurones,epochs=500):
    label = np.random.normal(0,1, [X_size,noise_size])
    X = generator.predict(label)
    encoder = create_encoder(neurones)
    # Fit the model
    encoder.fit(X, label, epochs=epochs)
    return encoder

#gan, generator = training(epochs=20)
encoder = train_encoder(generator, [20], 500)

label = np.random.normal(0,1, [50000,noise_size])
X = generator.predict(label)
label_pred = encoder.predict(X)
mu, std = norm.fit(label_pred)

x = np.linspace(min(label_pred),max(label_pred), 100)
p = norm.pdf(x, 0, 1)

plt.plot(x, p, 'k', linewidth=2)
title = "Gaussienne: mu = %.2f,  std = %.2f" % (mu, std)
plt.title(title)
plt.hist(label_pred, bins=40,density=True)
plt.show()

########################################
###  Import des modèles enregistrés  ###
########################################

def save_model(model, name):
    model_json = model.to_json()
    with open(name+".json", "w") as json_file:
        json_file.write(model_json)
    # enregistrement des poids
    model.save_weights(name+".h5")
    
save_model(gan, 'gan')
save_model(generator, 'generator')
save_model(encoder, 'encoder')
save_model(discriminator, 'gan')

def load_model(name):
    # chargement du réseau
    json_file = open(name+'.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # chargement des poids
    model.load_weights(name+'.h5')
    return model

#gan = load_model('gan')
#generator = load_model('generator')
#encoder = load_model('encoder')
#gan = load_model('gan')
