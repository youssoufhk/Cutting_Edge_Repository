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


tf.set_random_seed(2020)

default_path = "/Users/alixmathieu/Downloads/Cutting Edge/"
os.chdir(default_path)

#Reset all the graph
tf.reset_default_graph()

## Import our raw data
n_stocks = 50# nombre de stocks conservés
i = 0 # stock de départ
data_close = pd.read_csv('/Users/alixmathieu/Downloads/Cutting Edge/data.csv')
data_close = data_close.iloc[:,i:(i+n_stocks)]

n_time = data_close.shape[0]



# Data treatment ,fill the NA and calculate the daily reture
for j in range(0,n_stocks):
    data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())

#Data=(data_close)/np.std(data_close)


Data = np.log(data_close.values[1:,:]/data_close.values[:-1,:])

X_data_g =  Data*100 # 1/np.std(Data)
#X_data =  Data[:,np.random.randint(0,237,5)]*100

X_train_g, X_test_g = train_test_split(X_data_g, test_size = 0.3,shuffle=False)

###################################
############  STUDENT  ############
###################################
# Student 
NU_COPULA = 2
N_SIM = 1005
var = np.cov(np.transpose(X_train_g))

gauss_rv = multivariate_normal(cov=var).rvs(N_SIM).transpose()
chi2_rv = chi2(NU_COPULA).rvs(N_SIM)

mult_factor = np.sqrt(NU_COPULA / chi2_rv)

student_rvs = np.multiply(mult_factor, gauss_rv)
## student_rvs = vector R

inv_student_rvs = t.cdf(student_rvs, NU_COPULA).T

T = []
for i in range(N_SIM):
    T.append(t.ppf(inv_student_rvs[i,:],NU_COPULA))


X_data = np.array(T)
X_data
#Split the raw data to two part train and test
X_train, X_test = train_test_split(X_data, test_size = 0.3,shuffle=False)


Y_size = X_train.shape[0]     #the number of date we will used for one network training
X_size = X_train.shape[1]     #X_size is the number of stock
epochs = 4000
#the number of iteration

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=Y_size, dim=X_size):
    return np.random.uniform(-1,1,(n,dim))

def sample_noise_Gaus(n=Y_size, dim=X_size):
    return np.random.normal(0,1,(n,dim))

def sample_noise_multiGaus(n=Y_size):
    return np.random.multivariate_normal(np.zeros(5),np.eye(5),n)
##np.mean(X_train,axis=0)

def generator(Z,nb_neurone=[64,32,16],reuse=False):
    """ generator structure
        Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,nb_neurone[0],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,X_size)
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
        output = tf.layers.dense(h1,X_size)
    return output



X = tf.placeholder(tf.float32,[None,X_size])
Z = tf.placeholder(tf.float32,[None,X_size])


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




sess = tf.Session()
sess.run(tf.global_variables_initializer())
nd_steps=2#entrainer plus de dis que gen
ng_steps=1
d_loss_list=[]
g_loss_list = []



X_batch = X_train


## Boucle d'entraînement
with tf.device('/device:GPU:0'):
    for i in range(epochs):
        Z_batch = sample_noise_Gaus(Y_size,X_size)
        #ind_X = random.sample(range(Y_size),Y_size)
        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        
        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss],  feed_dict={X: X_batch, Z: Z_batch})
        
        if i%2==0:
            #print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)
            d_loss_list.append(dloss)
            g_loss_list.append(gloss)
        
        if i%100 ==0:
            print("Iteration :", i)



#Generate data with our generator by feeding the z generated by the encoder



## Plot les loss
def plot_loss(d_loss_list,g_loss_list):
    plt.subplot(2, 1, 1)
    plt.plot(d_loss_list, 'yo-')
    plt.ylabel('d_loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(g_loss_list,'r.-')
    plt.ylabel('g_loss')

#plot the loss
plt.figure(num=0, figsize=(7, 5))

plot_loss(d_loss_list,g_loss_list)


def KDE(X,Y):
    """ Evaluation function
        Args:
        X: The real data
        Y : The generated data
        Return:
        score: the score of generated data, higher the score is, better the data is
    """
    h = np.std(X_batch)*(4/3/len(X_batch))**(1/5) #0.5
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)
    score = sum(kde.score_samples(Y))
    return score

#Generate data with our generator by feeding Z
    
Z_batch = sess.run(z_sample,feed_dict={X: X_data_g})
pred = sess.run(gen_sample,feed_dict={Z: Z_batch}) 

#Z_test = sample_noise_Gaus(np.shape(X_test)[0],X_size)
#pred_test=sess.run(gen_sample,feed_dict={Z: Z_test}) 
#score = KDE(pred_test,X_test)


#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5
#Prob_real=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})
#Prob_pred=sess.run(real_output,feed_dict={X: pred,Z:Z_batch})
#Pred_test = sess.run(real_output,feed_dict={X:X_test,Z:Z_batch})


## Plot les data historique
def plot_historique(X_batch,pred):
    for i in range(10):
        plt.figure(num=1, figsize=(7, 5))
        D0 = pd.DataFrame(np.transpose((X_batch[:,i],pred[:,i])))
        D0.columns = ['real','fake']
        D0.plot.density()
        #plt.ylim((-0.05, 0.4))
        #plt.xlim((-2, 2))
        plt.title('return series of stock %s'%i)
        plt.axis([-50, 50, 0, 1])
plot_historique(X_data,pred)

print("The score of predition is :", score,"The Best score  is :",  KDE(pred_test,pred_test))


cov_originale = np.cov(np.transpose(X_data))
cov_pred = np.cov(np.transpose(pred))

mean_originale = np.mean(X_data,axis = 0)
mean_pred = np.mean(pred, axis = 0)


####################################
###### DETECTION D'ANOMALIE ########
####################################
# Renvoie le quantile du khi2 équivalent d'observations multidimensionnelles
def khi2_test(valeur,dim=X_size):
    khi2 = np.sum(np.multiply(valeur,valeur),axis=1)
    quantile = chi2.cdf(khi2,df=dim)
    return quantile

# Renvoie le nombre de valeurs détectées comme anomalies et leur quantile
def detection(valeurs, seuil=0.99, verbose=False):
    #dim = len(valeurs[0])
    n =  len(valeurs)
    label = sess.run(z_sample,feed_dict={X: valeurs})
    #khi2 = np.sum(np.multiply(label,label),axis=1)
    #quantile = chi2.cdf(khi2,df=dim)
    quantile = khi2_test(label)
    condition = [bool(i) for i in np.sum(([quantile > seuil],[quantile < 1-seuil]),axis=0)[0] ]
    indices = np.array(range(n))[condition]
    
    nb_anomalies = len(quantile[indices])
    if verbose:
        print("Nombre de quantiles > ",seuil,":",nb_anomalies,"(%.2f%%)"%(100*nb_anomalies/n))

    anomalies =[np.array(valeurs)[indices],np.array(quantile)[indices]]
    return anomalies, nb_anomalies

print("Avec rendements historiques")
_,_=detection(valeurs=X_test_g,seuil=0.99,verbose=True)

# Test de détection d'anomalie sur une observation modifiée de X_test
def test_detection(data, valeur, ligne=0,colonne=1, seuil=0.99, show_res = False, verbose=False):
    valeur_ini=[data[ligne,:]]
    valeur_mod = deepcopy(valeur_ini)
    valeur_mod[0][colonne] = valeur
    seuil = 0.99
    _,nb_ini = detection(valeur_ini, seuil, verbose)
    _,nb_mod = detection(valeur_mod, seuil, verbose)
    detecte = 0
    if nb_ini==0 and nb_mod == 1:
            detecte = 1
    
    if show_res:    
        if nb_ini==1:
            print("La valeur initiale a été détectée comme une anomalie.")
            if nb_mod==1:
                print("La valeur modifiée a aussi été détectée comme une anomalie.")
            if nb_mod==0:
                print("La valeur modifiée n'est plus détectée comme une anomalie.")
        if nb_ini==0 and nb_mod == 1:
            print("La valeur modifiée a été détectée comme une anomalie.")
        if nb_ini==0 and nb_mod == 0:
            print("La valeur modifiée n'a pas été détectée comme une anomalie.")
        
    return detecte

print("Avec rendements historiques")
_ = test_detection(X_test_g,5,show_res=True)

# Test du seuil de détection d'anomalie par colonne sur les observations de X_test
def recherche_seuil(data, colonne=0, ligne=0, seuil=0.99, nb_pas=50, val_max=5):
    val_test = np.linspace(0,val_max,nb_pas)
    detecte = 0
    nb=0

    while (detecte !=1 and nb<nb_pas):
        val = val_test[nb]
        detecte = test_detection(data, val, ligne=ligne, colonne=colonne,seuil=seuil)
        nb+=1
    if detecte==1:
        print("détection d'anomalie pour la valeur ",nb,": %.2f,"%val,"quantile %.2f%%"%(100*norm.cdf(val)))
    else:
        print("Aucune valeur seuil détectée <",val_max,"pour la colonne",colonne)
    return val
    
print("Avec rendements historiques")
for col in range(n_stocks):
    recherche_seuil(X_test_g, col, seuil=0.99, nb_pas=100, val_max=10)
    
    
# Générateur d'anomalie ligne à ligne
def anomalies(data_train,n=1, multiplicateur=5, seuil=0.99):
    var = multiplicateur*np.cov(np.transpose(data_train))
    svar = np.linalg.cholesky(var)
    mean = np.mean(data_train,axis=0)
    anomalies = []
    for i in range(n):
        non_extreme = True
        while non_extreme:
            anomalie = np.random.multivariate_normal(np.zeros(X_size),np.eye(X_size),1)
            
            quantile = khi2_test(anomalie)
            if quantile > seuil:
                non_extreme = False
        anomalie = np.dot(svar,anomalie[0])+mean
        anomalies.append(anomalie)
    return anomalies

print("Avec rendements historiques")
test_ano1 = anomalies(X_train_g,100,5)
_,_=detection(valeurs=test_ano1,seuil=0.99,verbose=True)


# Générateur d'anomalies par paquets
def anomalies_mult(data_train, n=1, multiplicateur=5, seuil=0.99):
    var = multiplicateur*np.cov(np.transpose(data_train))
    svar = np.linalg.cholesky(var)
    mean = np.mean(data_train,axis=0)
    nb_stock = data_train.shape[1]
    anomalies = []
    
    while len(anomalies)<n:
        valeurs = np.random.multivariate_normal(np.zeros(X_size),np.eye(X_size),int(nb_stock*n))
        quantile = khi2_test(valeurs)
        condition = tuple([quantile > seuil])
        indices = np.array(range(int(nb_stock*n)))[condition]
        new_anomalies = np.array(valeurs)[indices]
        anomalies.extend(new_anomalies)
    
    anomalies = np.array(anomalies[:n])
    anomalies = np.dot(anomalies,svar.T)+mean
    return anomalies

# Détection des anomalies multivariées générées
def test_anomalies_mult(data_train, n=1, multiplicateur=5, seuil=0.99,verbose=True):
    anomalies = anomalies_mult(data_train,n,multiplicateur,seuil)
    _,nb=detection(valeurs=anomalies,seuil=0.99,verbose=False)
    if verbose:
        print("Nombre de quantiles > ",seuil,"multiplicateur : %.2f"%multiplicateur,":",nb,"(%.2f%%)"%(100*nb/n))
    return anomalies
    
print("Avec rendements historiques")
for mult in np.linspace(.1,2.,50):
    _ = test_anomalies_mult(X_train_g,10000,mult)

