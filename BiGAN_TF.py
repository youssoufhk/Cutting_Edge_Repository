#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 17:20:16 2020

@author: Yousshk
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.model_selection import train_test_split

#Reset all the graph
tf.reset_default_graph()

## Import our raw data
data_close = pd.read_csv('C:/Users/Yousshk/Documents/CuttingEdge/data.csv')

n_time = data_close.shape[0]
n_stocks = data_close.shape[1]


# Data treatment, fill the NA and calculate the daily reture
for j in range(0,n_stocks):
        data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())
rendements = np.log(data_close.values[1:,:]/data_close.values[:-1,:])
rendements = np.float32(rendements)

#Split the raw data to two part train and test
X_train, X_test = train_test_split(rendements, test_size = 0.33, random_state = 42)

#Constants declaration
X_size = X_train.shape[1] #X_size is the number of stock
ntime = X_train.shape[0]
noise_size = X_size
batch_size = 32               #the number of date we will used for one network training
nb_batch = int(X_size/ batch_size)
epochs = 100             #the number of iteration

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=batch_size, c_size=noise_size):
    return np.random.uniform(-1,1,(n,c_size))

def sample_noise_Gaus(n=batch_size, c_size=noise_size):
    return np.random.normal(-1,1,(n,c_size))


def generator(Z,nb_neurone=100,reuse=False):
    """ generator structure
    Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
    """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,nb_neurone,activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,nb_neurone,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h2,X_size)
    return output


def discriminator(X,Z,nb_neurone=100,reuse=False):
    """ discriminator structure
    Args:
        X: The real data or generated data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
    """
    input = tf.concat( axis=1,values=[X,Z])
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(input,nb_neurone,activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,nb_neurone,activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,2)
        output = tf.layers.dense(h3,1)

    return output


def encoder(X,nb_neurone=100,reuse=False):
    """ encoder structure
    Args:
        X: The real data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
    """
    with tf.variable_scope("GAN/Encoder",reuse=reuse):
        h1 = tf.layers.dense(X,nb_neurone,activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,nb_neurone,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h2,noise_size)
    return output



X = tf.placeholder(tf.float32,[None,X_size])
Z = tf.placeholder(tf.float32,[batch_size,noise_size])


gen_sample = generator(Z)
z_sample = encoder(X)
corr = tf.placeholder(tf.float32,[batch_size-1])
real_output = discriminator(X, z_sample)
fake_output = discriminator(gen_sample, Z,reuse=True)


#Discriminator loss
disc_loss = -tf.reduce_mean(tf.log(real_output+1e-7) + tf.log(1.0-fake_output+1e-7))

#Generator loss
gen_loss = -tf.reduce_mean(tf.log(fake_output+1e-7) + tf.log(1.0-real_output+1e-7))


#Define the Optimizer with learning rate  0.001

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")

disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Encoder")

gen_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars+enc_vars)
disc_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars)


#gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars+enc_vars)
#disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars)



sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Training process
with tf.device('/device:GPU:0'):
    for i in range(epochs):
        ind_X = random.sample(range(ntime),ntime)
        for j in range(1,nb_batch+1):
            index = ind_X[(j-1)*batch_size:j*batch_size]
            X_batch = X_train[index]
            Z_batch = sample_noise_Gaus(batch_size, noise_size)
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
        if i%100==0:
            print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)



#Generate data with our generator by feeding Z
test = generator(z_sample)
pred=sess.run(test,feed_dict={Z: Z_batch})

#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5

y_real=sess.run(real_output,feed_dict={X: X_batch})

Prob_real=sess.run(tf.sigmoid(y_real))

y_pred=sess.run(real_output,feed_dict={X: pred})

Prob_pred=sess.run(tf.sigmoid(y_pred))


#Check the cov matrix (problem need to solve)
Cov_pred = np.corrcoef(np.transpose(pred))
#
Cov_X = np.corrcoef(np.transpose(X_batch))
