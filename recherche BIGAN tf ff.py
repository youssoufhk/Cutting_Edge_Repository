#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
import os

tf.set_random_seed(2020)

default_path = "D:/Documents/Cours/Cutting Edge/"
os.chdir(default_path)

tf.reset_default_graph()

#n = 500 # nombre d'obervations

## Génération des vrais datas
#np.random.seed(7)
#cov = np.array([[.1, .5],[.5, .1]])
#data = np.random.multivariate_normal(mean=[2,2], cov= cov, size=n)

nb_stock=5

data = pd.read_csv('data.csv')
data=np.array(data)
data = data[:,0:nb_stock]

n_time = data.shape[0]

## Creation du dataframe des rendements
rendements = 100*np.log(data[1:]/data[:-1])
# Découpage en échantillon d'apprentissage et de test
X_train, X_test = train_test_split(rendements, test_size = 0.33, random_state = 42)

# Initialisation des paramètres
X_size = X_train.shape[1]
ntime = X_train.shape[0]
noise_size = 1
batch_size = 50
nb_batch = int(ntime/ batch_size) 
reste = ntime-batch_size*nb_batch
nb_batch = nb_batch + 1*(reste>0)
    
    
plt.plot(range(len(X_train)),X_train)
##############################################
##############################################
def sample_noise_uniform(n=batch_size, c_size=noise_size):        
    return np.random.uniform(-1,1,(n,c_size))

def sample_noise_Gaus(n=batch_size, c_size=noise_size):        
    return np.random.normal(-1,1,(n,c_size))

 
def generator(noise ,nb_neurone=[10],couches=1,reuse=False):
    """ generator structure
    Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(noise, units=nb_neurone[0], activation=tf.nn.leaky_relu)
        for i in range(couches-1):
            h1 = tf.layers.dense(h1, units=nb_neurone[i+1], activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1, X_size)
    return output
        

def discriminator(X,nb_neurone=[10],couches=1,reuse=False):
    """ discriminator structure
    Args:
        X: The real data or generated data 
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X, units=nb_neurone[0], activation=tf.nn.leaky_relu)
        for i in range(couches-1):
            h1 = tf.layers.dense(h1, units=nb_neurone[i+1], activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1, units=1)
    return output

def encoder(X,nb_neurone=[10],couches=1,reuse=False):
    """ encoder structure
        Args:
        X: The real data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("BiGAN/Encoder",reuse=reuse):
        h1 = tf.layers.dense(X, units=nb_neurone[0], activation=tf.nn.leaky_relu)
        for i in range(couches-1):
            h1 = tf.layers.dense(h1, units=nb_neurone[i+1], activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,noise_size)
    return output

def KDE(x,y):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x)
    score = sum(kde.score_samples(y[:,]))
    return score
KDE(X_train,X_test)


def score(neurones,epoques=2000,ng=1,nd=1,lr=0.001,verbose=False):
    # Résultat des scores de chaque modèle
    scores = [] 
    for nb_neur in neurones: 
        couches = len(nb_neur)
        tf.reset_default_graph()
        X = tf.placeholder(tf.float32,[batch_size,X_size])
        noise = tf.placeholder(tf.float32,[batch_size,noise_size])
        
        # Générateur 
        Generateur = generator(noise,nb_neur,couches)
        # Discriminateur
        Discriminateur_vrai = discriminator(X,nb_neur,couches)
        Discriminateur_faux = discriminator(Generateur,nb_neur,couches,reuse=True)
        
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
        
        
        #Define the Optimizer with learning rate  lr
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
        enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BiGAN/Encoder")
        #gen_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(gen_loss,var_list = gen_vars) 
        #disc_step = tf.train.AdamOptimizer(learning_rate=lr).minimize(disc_loss,var_list = disc_vars) 
        gen_step = tf.train.RMSPropOptimizer(learning_rate=lr,decay=0.9, momentum=0.0, epsilon=1e-8).minimize(gen_loss,var_list = gen_vars+ enc_vars) 
        disc_step = tf.train.RMSPropOptimizer(learning_rate=lr,decay=0.9, momentum=0.0, epsilon=1e-8).minimize(disc_loss,var_list = disc_vars)
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        #Training process
        with tf.device('/device:GPU:0'):
            for i in range(epoques): # boucle sur les époques
                # Tirage d'un ordre aléatoire sur X
                ind_X = random.sample(range(ntime),ntime)
                for j in range(nb_batch): # boucle sur les différents batchs (n/batch_size)
                    if(j < nb_batch-1):
                        index = ind_X[j*batch_size:(j+1)*batch_size]
                        Z_batch = sample_noise_uniform(batch_size, noise_size) # bruit uniforme
                    else:
                        index = ind_X[(ntime-batch_size):]
                        Z_batch = sample_noise_uniform(batch_size, noise_size) # bruit uniforme
                    X_batch = X_train[index,:]
                    
                    X_batch = X_batch.reshape(batch_size,X_size)
                    Z_batch = Z_batch.reshape(batch_size,noise_size)
                    
                    for _ in range(nd):
                        _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, noise: Z_batch})
                    for _ in range(ng):
                        _, gloss = sess.run([gen_step, gen_loss], feed_dict={noise: Z_batch})
        
                if ((i+1)%100==0) & verbose:
                    print ("Iteration:",i+1, "Discriminator loss: ", dloss, "Generator loss:", gloss)

        
        # Affiche la moyenne et la vol estimée, et la valeur moyenne du dis sur les données
        # Génération de ntime rendements 
        bruits = sample_noise_uniform(batch_size, noise_size)
        pred=sess.run(Generateur,feed_dict={noise: bruits})
        # Score du modèle entrainé
        score = KDE(X_test,pred)
        print("Score pour C:",couches," N:",nb_neur,", score:",score)
        
        # Affichage complémentaire
        if verbose:
            # Selection d'un batch de données observées
            index = random.sample(range(ntime),batch_size)
            X_batch = X_train[index].reshape(batch_size,X_size)
            # Vérification du discriminateur
            prob_observe = sess.run(Discriminateur_vrai,feed_dict={X: X_batch})
            prob_observe = tf.nn.sigmoid(prob_observe)
            prob_genere = sess.run(Discriminateur_vrai,feed_dict={X: pred})
            prob_genere = tf.nn.sigmoid(prob_genere)    
            print("moyenne Discriminateur sur données observées :",np.mean(prob_observe))
            print("moyenne Discriminateur sur données générées :",np.mean(prob_genere))
            
            # Comparaison des distributions observées et estimées
            
            Mean_pred = np.mean(np.transpose(pred),axis=1)
            Mean_X = np.mean(np.transpose(X_batch),axis=1)
            
            print("vrai moyenne:",Mean_X,"\nmoyenne estimée:",Mean_pred)
            print('pred:',pred)
            Cov_pred = np.cov(np.transpose(pred))
            Cov_X = np.cov(np.transpose(X_batch))
            
            print("cov predite:",Cov_pred)
            print("cov observée:",Cov_X)
            
            plt.hist(pred,density=1,edgecolor='blue',bins=30)
            plt.hist(X_batch,density=1,edgecolor='red',bins=30)
            plt.show()
            
            pd.DataFrame(np.transpose((X_batch[:,0],pred[:,0])),columns = ['real','fake']).plot.density()
        nom = "C"+str(couches)+"N"+str(nb_neur)+";"
        tmp = [nom,score]
        scores.append(tmp)
    return scores

modeles = [[10],[10,10],[10,10,10],[10,10,10,10],[10,10,10,10,10],
[20],[20,20],[20,20,20],[20,20,20,20],[20,20,20,20,20],
[30],[30,30],[30,30,30],[30,30,30,30],[30,30,30,30,30],
[40],[40,40],[40,40,40],[40,40,40,40],[40,40,40,40,40],
[50],[50,50],[50,50,50],[50,50,50,50],[50,50,50,50,50]]

scores = score(modeles,epoques=100,ng=1,nd=1,lr=0.001,verbose=False)
pd.DataFrame(scores).to_csv('scores.csv',index = False)