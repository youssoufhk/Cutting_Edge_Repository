#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:05:19 2020
@author: Feniza
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

n = 500 # nombre d'obervations

## Génération des vrais datas
#np.random.seed(7)
#cov = np.array([[.1, .5],[.5, .1]])
#data = np.random.multivariate_normal(mean=[2,2], cov= cov, size=n)

data = pd.read_csv('data.csv')
data=np.array(data)
data = data[:,0]
n_time = data.shape[0]
n_stocks = 1

#Split the raw data to two part train and test
X_train, X_test = train_test_split(data, test_size = 0.33, random_state = 42)

#Constants declaration 
X_size = 1 #X_size is the number of stock
ntime = X_train.shape[0]
noise_size = 1
batch_size = 50               #the number of date we will used for one network training
nb_batch = int(X_size/ batch_size) 
reste = X_size-batch_size*nb_batch
nb_batch = nb_batch + 1*(reste>0)
epochs = 5000             #the number of iteration

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=batch_size, c_size=noise_size):        
    return np.random.uniform(-1,1,(n,c_size))

def sample_noise_Gaus(n=batch_size, c_size=noise_size):        
    return np.random.normal(-1,1,(n,c_size))

 
def generator(noise ,nb_neurone=10,reuse=False):
    """ generator structure
    Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(noise, units=nb_neurone, activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1, X_size)
    return output


def discriminator(X,nb_neurone=10,reuse=False):
    """ discriminator structure
    Args:
        X: The real data or generated data 
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X, units=nb_neurone, activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1, units=1, activation=tf.nn.sigmoid)
    return output

X = tf.placeholder(tf.float32,[batch_size,X_size])
noise = tf.placeholder(tf.float32,[batch_size,noise_size])

corr = tf.placeholder(tf.float32,[batch_size-1])

# Générateur 
Generateur = generator(noise)
# Discriminateur
Discriminateur_vrai = discriminator(X)
Discriminateur_faux = discriminator(Generateur,reuse=True)

#corr = tf.transpose(tfp.stats.correlation(gen_sample))
#corr_loss = tf.reduce_sum(corr)- tf.reduce_sum(tf.diag_part(corr))

#dis_loss = min E[-log(D(X))] + E[log(1-D(G(Z)))] := real_loss + gen_loss
#sigmoid_cross_entropy_with_logits(x,z) = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
label_dis_vrai = 0.9*tf.ones_like(Discriminateur_vrai)
label_dis_faux = tf.zeros_like(Discriminateur_faux)
label_gen=tf.ones_like(Discriminateur_faux) # on fait croire au dis que la sortie du gen est vraie

dis_r_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=Discriminateur_vrai,labels=label_dis_vrai)
dis_f_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=Discriminateur_faux,labels=label_dis_faux)
disc_loss = tf.reduce_mean(dis_r_loss + dis_f_loss)

#gen_loss = min E[log(1-D(G(Z)))] =  max  E[log D(G(Z)] = min - E[log(D(G(Z)))]
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Discriminateur_faux,labels=label_gen))


#Define the Optimizer with learning rate  0.001
gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

gen_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) 
disc_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) 
#gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars) 
#disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars) 



sess = tf.Session()
sess.run(tf.global_variables_initializer())

#Training process
with tf.device('/device:GPU:0'):
    for i in range(epochs): # boucle sur les époques
        # Tirage d'un ordre aléatoire sur X
        ind_X = random.sample(range(ntime),ntime)
        for j in range(nb_batch): # boucle sur les différents batchs (n/batch_size)
            if(j < nb_batch):
                index = ind_X[j*batch_size:(j+1)*batch_size]
                Z_batch = sample_noise_uniform(batch_size, noise_size) # bruit uniforme
            else:
                index = ind_X[j*batch_size:]
                Z_batch = sample_noise_uniform(reste, noise_size) # bruit uniforme
            X_batch = X_train[index]
            
            X_batch = X_batch.reshape(batch_size,X_size)
            Z_batch = Z_batch.reshape(batch_size,noise_size)
            
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, noise: Z_batch})
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={noise: Z_batch})

        if (i+1)%100==0:
            print ("Iteration:",i+1, "Discriminator loss: ", dloss, "Generator loss:", gloss)
        

#Generate data with our generator by feeding Z 
pred=sess.run(Generateur,feed_dict={noise: Z_batch})

#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5
y_real=sess.run(Discriminateur_vrai,feed_dict={X: X_batch})

Prob_real=sess.run(tf.sigmoid(y_real))

y_pred=sess.run(Discriminateur_vrai,feed_dict={X: pred})

Prob_pred=sess.run(y_pred)


#Check if the Cov and mean are good
np.set_printoptions(suppress=True)

Mean_pred = np.mean(np.transpose(pred),axis=1)
Mean_X = np.mean(np.transpose(X_batch),axis=1)
Cov_pred = np.around(np.cov(np.transpose(pred)), decimals=3)
#print(np.around(np.cov(np.transpose(pred)), decimals=2))
Cov_X = np.around(np.cov(np.transpose(X_batch)), decimals=3)
#print(np.around(np.cov(np.transpose(X_batch)), decimals=2))
