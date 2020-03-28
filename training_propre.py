#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm,multivariate_normal, chi2, t
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from copy import deepcopy




default_path = "/Users/alixmathieu/Downloads/Cutting Edge/"
os.chdir(default_path)

# Reset all the graph
tf.reset_default_graph()
tf.set_random_seed(2020)

epochs = 2000

## Import our raw data
n_stocks = 5 # number of stocks to consider
i = 0

data_close = pd.read_csv('/Users/alixmathieu/Downloads/Cutting Edge/data.csv')
data_close = data_close.iloc[:,i:(i+n_stocks)]

# Data treatment ,fill the NA by the mean
for j in range(0,n_stocks):
    data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())

# Take the daily returns
Data = np.log(data_close.values[1:,:]/data_close.values[:-1,:])

X_data =  Data*100 # 1/np.std(Data)
X_train, X_test = train_test_split(X_data, test_size = 0.3,shuffle=False)
n_time = X_train.shape[0]


##############################################
## Generate noise
def sample_noise_uniform(n=n_time, dim=n_stocks):
    return np.random.uniform(-1,1,(n,dim))

def sample_noise_Gaus(n=n_time, dim=n_stocks):
    return np.random.normal(0,1,(n,dim))

def sample_noise_multiGaus(n=n_time,dim = n_stocks):
    return np.random.multivariate_normal(np.zeros(dim),np.eye(dim),n)


## Implementation of the 3 networks
def generator(Z,nb_neurone=[64,32,16],reuse=False):
    """ generator structure
        Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,nb_neurone[0],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,n_stocks)
    return output


def discriminator(X,Z,nb_neurone=[128,64,32],reuse=False):
    """ discriminator structure
        Args:
        Z:
        X: The real data or generated data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    input = tf.concat((X,Z),1)
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(input,nb_neurone[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,nb_neurone[1],activation=tf.nn.leaky_relu)
        h3 = tf.layers.dense(h2,nb_neurone[2])
        output = tf.layers.dense(h3,1,activation=tf.nn.sigmoid)
    return output


def encoder(X,nb_neurone=[64,32,16],reuse=False):
    """ encoder structure
        Args:
        X: The real data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("GAN/Encoder",reuse=reuse):
        h1 = tf.layers.dense(X,nb_neurone[0],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,n_stocks)
    return output



X = tf.placeholder(tf.float32,[None,n_stocks])
Z = tf.placeholder(tf.float32,[None,n_stocks])


gen_sample = generator(Z)
z_sample = encoder(X)

real_output = discriminator(X, z_sample)
fake_output = discriminator(gen_sample, Z,reuse=True)


#Discriminator loss
disc_loss = -tf.reduce_mean(tf.log(real_output+1e-10) + tf.log(1.0-fake_output+1e-10))

#Generator loss
gen_loss = -tf.reduce_mean(tf.log(fake_output+1e-10) + tf.log(1.0-real_output+1e-10))


#Define the Optimizer with learning rate  0.001
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")

disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Encoder")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.5, momentum=0.0).minimize(gen_loss,var_list = gen_vars+ enc_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.5, momentum=0.0).minimize(disc_loss,var_list = disc_vars)



# Training of the BiGAN

sess = tf.Session()
sess.run(tf.global_variables_initializer())
nd_steps=2 #entrainer plus de dis que gen
ng_steps=1
d_loss_list=[]
g_loss_list = []

X_batch = X_train


## Training loop
with tf.device('/device:GPU:0'):
    for i in range(epochs):
        Z_batch = sample_noise_Gaus(n_time,n_stocks)
        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        
        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss],  feed_dict={X: X_batch, Z: Z_batch})
        
        if i%2==0:
            d_loss_list.append(dloss)
            g_loss_list.append(gloss)
        
        if i%100 ==0:
            print("Iteration :", i)


## Plot of the loss functions during training
def plot_loss(d_loss_list,g_loss_list):
    plt.subplot(2, 1, 1)
    plt.plot(d_loss_list, 'yo-')
    plt.ylabel('d_loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(g_loss_list,'r.-')
    plt.ylabel('g_loss')

plt.figure(num=0, figsize=(7, 5))
plot_loss(d_loss_list,g_loss_list)



#Generate data with our generator by feeding Z

Z_batch = sess.run(z_sample,feed_dict={X: X_batch})
pred = sess.run(gen_sample,feed_dict={Z: Z_batch}) 

#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5

Prob_real=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})
Prob_pred=sess.run(real_output,feed_dict={X: pred,Z:Z_batch})


## Compare distributions of each stock (original/generated)
def plot_distribution(X_batch,pred):
    for i in range(X_batch.shape[1]):
        plt.figure(num=1, figsize=(7, 5))
        D0 = pd.DataFrame(np.transpose((X_batch[:,i],pred[:,i])))
        D0.columns = ['real','fake']
        D0.plot.density()
        plt.title('return series of stock %s'%i)
        plt.axis([-50, 50, 0, 1])
plot_distribution(X_batch,pred)

## Compare mean and covariance matrix for the original data and the generated data
original_cov = np.cov(np.transpose(X_batch))
predicted_cov = np.cov(np.transpose(pred))

original_mean = np.mean(X_data,axis = 0)
predicted_mean = np.mean(pred, axis = 0)
