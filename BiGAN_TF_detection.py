#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2, norm, multivariate_normal, t
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import os
from sklearn import preprocessing
from copy import deepcopy
 
 
tf.set_random_seed(2020)
np.random.seed(7)
default_path = "D:/Documents/Cours/Cutting Edge/"
os.chdir(default_path)

#Reset all the graph
tf.reset_default_graph()
## Import our raw data
n_stocks = 5 # nombre de stocks conservés
i = 0 # stock de départ
data_close = pd.read_csv('data.csv')
data_close = data_close.iloc[:,i:(i+n_stocks)]

n_time = data_close.shape[0]



# Data treatment ,fill the NA and calculate the daily reture
for j in range(0,n_stocks):
    data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())

#Data=(data_close)/np.std(data_close)


Data = np.log(data_close.values[1:,:]/data_close.values[:-1,:])

X_data =  Data*100 # 1/np.std(Data)
#X_data =  Data[:,np.random.randint(0,237,5)]*100

#Split the raw data to two part train and test


X_train, X_test = train_test_split(X_data, test_size = 0.35,shuffle=False)
#X_train, X_test = train_test_split(X_data, test_size = 0.35,random_state=42)
#Constants declaration
Y_size = X_train.shape[0]     #the number of date we will used for one network training
X_size = X_train.shape[1]     #X_size is the number of stock
epochs = 2000
#the number of iteration

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=Y_size, dim=X_size):
    return np.random.uniform(-1,1,(n,dim))

def sample_noise_Gaus(n=Y_size, dim=X_size):
    var = np.var(X_train)
    mean = np.mean(X_train)
    return np.random.normal(0,1,(n,dim))

def sample_noise_multiGaus(n=Y_size):
    var = np.cov(np.transpose(X_train))
    mean = np.mean(X_train,axis=0)
    return np.random.multivariate_normal(mean,var,n)


def generator(Z,nb_neurone=[64,32],reuse=False):
    """ generator structure
        Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,nb_neurone[0],activation=tf.nn.leaky_relu)
        h2 = tf.layers.dense(h1,nb_neurone[1],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h2,X_size)
    return output


def discriminator(X,Z,nb_neurone=[64,32],reuse=False):
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
        h3 = tf.layers.dense(h2,2)
        output = tf.layers.dense(h3,1,activation=tf.nn.sigmoid)
    return output


def encoder(X,nb_neurone=[64,32],reuse=False):
    """ encoder structure
        Args:
        X: The real data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("GAN/Encoder",reuse=reuse):
        h1 = tf.layers.dense(X,nb_neurone[0],activation=tf.nn.leaky_relu)
        #h2 = tf.layers.dense(h1,32,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,X_size)
    return output



X = tf.placeholder(tf.float32,[None,X_size])
Z = tf.placeholder(tf.float32,[None,X_size])


gen_sample = generator(Z)
z_sample = encoder(X)

real_output = discriminator(X, z_sample)
fake_output = discriminator(gen_sample, Z,reuse=True)


#Discriminator loss
disc_loss = -tf.reduce_mean(tf.log(real_output+1e-5) + tf.log(1.0-fake_output+1e-5))

#Generator loss
gen_loss = -tf.reduce_mean(tf.log(fake_output+1e-5) + tf.log(1.0-real_output+1e-5))


#Define the Optimizer with learning rate  0.001

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")

disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")

enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Encoder")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars+ enc_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars)




sess = tf.Session()
sess.run(tf.global_variables_initializer())
nd_steps=1#entrainer plus de dis que gen
ng_steps=1
d_loss_list=[]
g_loss_list = []

X_batch = X_train
with tf.device('/device:GPU:0'):
    for i in range(epochs):
        Z_batch = sample_noise_Gaus(Y_size,X_size)
        #ind_X = random.sample(range(Y_size),Y_size)
        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        
        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss],  feed_dict={X: X_batch, Z: Z_batch})
        
        if i%100==0:
            print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)
            d_loss_list.append(dloss)
            g_loss_list.append(gloss)




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
        score: the score of generated data, the higer score is, the better data is
        """
    h = np.std(X_batch)*(4/3/len(X_batch))**(1/5) #0.5
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)
    score = sum(kde.score_samples(Y))
    return score

#Generate data with our generator by feeding Z

#Z_batch = sess.run(z_sample,feed_dict={X: X_batch})

#test = generator(z_sample,reuse=True)

Z_batch = sess.run(z_sample,feed_dict={X: X_batch})


pred=sess.run(gen_sample,feed_dict={Z: Z_batch})

print("The score of predition is :", KDE(X_batch,pred),"The best score is :", KDE(X_batch,X_batch))

#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5
"""
    y_real=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})
    
    Prob_real=sess.run(y_real)
    
    y_pred=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})
    
    Prob_pred=sess.run(y_real)
    """

#Check if the Cov and mean are good
np.set_printoptions(suppress=True)

Mean_pred = np.mean(np.transpose(pred),axis=1)
Mean_X = np.mean(np.transpose(X_batch),axis=1)
Cov_pred = np.around(np.cov(np.transpose(pred)), decimals=3)
#print(np.around(np.cov(np.transpose(pred)), decimals=2))
Cov_X = np.around(np.cov(np.transpose(X_batch)), decimals=3)
#print(np.around(np.cov(np.transpose(X_batch)), decimals=2))

Corr_pred = np.around(np.corrcoef(np.transpose(pred)), decimals=3)
Corr_X = np.around(np.corrcoef(np.transpose(X_batch)), decimals=3)

plt.figure(num=1, figsize=(7, 5))

D0 = pd.DataFrame(np.transpose((X_batch[:,0],pred[:,0])))
D0.columns = ['real','fake']
D0.plot.density()
#plt.ylim((-0.05, 0.4))
#plt.xlim((-25, 25))
plt.title('return series of stock 1')

"""
    plt.figure(num=2, figsize=(7, 5))
    D1 = pd.DataFrame(np.transpose((X_batch[:,1],pred[:,1])))
    D1.columns = ['real','fake']
    D1.plot.density()
    #plt.ylim((-0.05, 0.4))
    #plt.xlim((-25, 25))
    plt.title('return series of stock 2')
    plt.show()
    
    plt.figure(num=3, figsize=(7, 5))
    D2 = pd.DataFrame(np.transpose((X_batch[:,2],pred[:,2])))
    D2.columns = ['real','fake']
    D2.plot.density()
    #plt.ylim((-0.05, 0.4))
    #plt.xlim((-25, 25))
    plt.title('return series of stock 3')
    plt.show()
    
    D3 = pd.DataFrame(np.transpose((X_batch[:,3],pred[:,3])))
    D3.columns = ['real','fake']
    D3.plot.density()
    #plt.ylim((-0.05, 0.4))
    #plt.xlim((-25, 25))
    plt.title('return series of stock 4')
    plt.show()
    
    D4 = pd.DataFrame(np.transpose((X_batch[:,4],pred[:,4])))
    D4.columns = ['real','fake']
    D4.plot.density()
    #plt.ylim((-0.05, 0.4))
    #plt.xlim((-25, 25))
    plt.title('return series of stock 5')
    plt.show()
    """

#n = 100*nb_stock # nombre d'obervations

## Génération des vrais datas
#np.random.seed(7)
#data = np.random.normal(size=n)
#data = np.resize(data,(int(n/nb_stock),nb_stock))

# Renvoie le quantile du khi2 équivalent d'observations multidimensionnelles
def khi2_test(valeur,dim=X_size):
    khi2 = np.sum(np.multiply(valeur,valeur),axis=1)
    quantile = chi2.cdf(khi2,df=dim)
    return quantile

# Renvoie le nombre de valeurs détectées comme anomalies et leur quantile
def detection_simple(valeurs, seuil=0.99, verbose=False):
    #dim = len(valeurs[0])
    n =  len(valeurs)
    var = np.cov(np.transpose(X_train))
    svar = np.linalg.inv(np.linalg.cholesky(var))
    mean = np.mean(X_train,axis=0)
    valeurs = np.dot(valeurs-mean,svar)
    
    quantile = khi2_test(valeurs)
    indices = [i for i in range(n) if quantile[i]>seuil or quantile[i] < 1-seuil]
    nb_anomalies = len(quantile[indices])
    if verbose:
        print("Nombre de quantiles > ",seuil,":",nb_anomalies,"(%.2f%%)"%(100*nb_anomalies/n))

    anomalies =[np.array(valeurs)[indices],np.array(quantile)[indices]]
    return anomalies, nb_anomalies

_,_ = detection_simple(X_test,verbose=True)

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
    indices = [i for i in range(n) if quantile[i]>seuil or quantile[i] < 1-seuil]
    nb_anomalies = len(quantile[indices])
    if verbose:
        print("Nombre de quantiles > ",seuil,":",nb_anomalies,"(%.2f%%)"%(100*nb_anomalies/n))

    anomalies =[np.array(valeurs)[indices],np.array(quantile)[indices]]
    return anomalies, nb_anomalies

_,_ = detection(valeurs=X_test,seuil=0.99,verbose=True)

# Test de détection d'anomalie sur une observation modifiée de X_test
def test_detection(valeur, ligne=0,colonne=0, seuil=0.99, show_res = False, verbose=False):
    valeur_ini=[X_test[ligne,:]]
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


_ = test_detection(2,show_res=True)

# Test du seuil de détection d'anomalie par colonne sur les observations de X_test
def recherche_seuil(colonne=0, ligne=0, seuil=0.99, nb_pas=50, val_max=5):
    val_test = np.linspace(0,val_max,nb_pas)
    detecte = 0
    nb=0

    while (detecte !=1 and nb<nb_pas):
        val = val_test[nb]
        detecte = test_detection(val, ligne=ligne, colonne=colonne,seuil=seuil)
        nb+=1
    if detecte==1:
        print("détection d'anomalie pour la valeur ",nb,": %.2f,"%val,"quantile %.2f%%"%(100*norm.cdf(val)))
    else:
        print("Aucune valeur seuil détectée <",val_max,"pour la colonne",colonne)
    return val
    
for col in range(n_stocks):
    recherche_seuil(col, seuil=0.99, nb_pas=50, val_max=10)
    
# Générateur d'anomalie ligne à ligne
def anomalies(n=1, multiplicateur=5, seuil=0.99):
    var = multiplicateur*np.cov(np.transpose(X_train))
    svar = np.linalg.cholesky(var)
    mean = np.mean(X_train,axis=0)
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


test_ano = anomalies(100,5)
_,_=detection(valeurs=test_ano,seuil=0.99,verbose=True)

# Générateur d'anomalies par paquets
def anomalies_mult(n=1, mult_var=5, mult_moy=1,seuil=0.99):
    var = mult_var*np.cov(np.transpose(X_train))
    mean = mult_moy*np.mean(X_train,axis=0)
    svar = np.linalg.cholesky(var)   
        
    anomalies = []
    while len(anomalies)<n:
        valeurs = np.random.multivariate_normal(np.zeros(X_size),np.eye(X_size),5*n)
        quantile = khi2_test(valeurs)
        condition = tuple([quantile > seuil])
        indices = np.array(range(5*n))[condition]
        new_anomalies = np.array(valeurs)[indices]
        anomalies.extend(new_anomalies)
    
    anomalies = np.array(anomalies[:n])
    anomalies = np.dot(anomalies,svar.T)+mean
    return anomalies


test_ano = anomalies_mult(1000,1)
_,_=detection(valeurs=test_ano,seuil=0.99,verbose=True)
_,_=detection_simple(valeurs=test_ano,seuil=0.99,verbose=True)



# Détection des anomalies multivariées générées
def test_anomalies_mult(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_mult(n,mult_var,mult_moy,seuil)
    _,nb=detection(valeurs=anomalies,seuil=0.99,verbose=False)
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(100*nb/n))
    return anomalies
    
for mult in np.linspace(.1,2.5,20):
    _ = test_anomalies_mult(n=10000,mult_var=mult,seuil=0.99)
    
for mult in np.linspace(1,45,30):
    _ = test_anomalies_mult(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    
anomalies = anomalies_mult(1000,20,1,seuil=.99)
print(np.mean(anomalies,axis=0))
_,nb=detection(valeurs=anomalies,seuil=0.99,verbose=True)
    
def mult_student(nb=1,mult_var=1,mult_moy=1,verbose=False):
    var = mult_var*np.cov(np.transpose(X_train))
    mean = mult_moy*np.mean(X_train,axis=0)
    NU_COPULA = X_size    
    
    gauss_rv = multivariate_normal(mean=mean,cov=var).rvs(nb).transpose()
    chi2_rv = chi2(NU_COPULA).rvs(nb)
    mult_factor = np.sqrt(NU_COPULA / chi2_rv)
    student_rvs = np.multiply(mult_factor, gauss_rv).T
    if verbose:
        for i in range(X_size):
            plt.hist(student_rvs[:,i],density=True)
            plt.show()
    return student_rvs

# quantile historique multi dimension
def quant_hist(x,alpha,verbose=False):
    khi2 = np.sum(np.multiply(x,x),axis=1)
    khi2.sort()
    quantile = khi2[int(alpha*len(khi2))]
    if verbose:
        print("quantile pour alpha=%f"%alpha,":",quantile)
    return quantile

# Seuil alpha pour une student multivariée
def seuil_student(alpha=0.99,mult_var=1,mult_moy=1,rep=100,nb=100000):
    seuil=0
    for i in range(rep):
        valeurs = mult_student(nb,mult_var,mult_moy,verbose=False)
        seuil += quant_hist(valeurs,.99,False)
    seuil=seuil/rep

    return seuil

# Générateur d'anomalies de Student par paquets
def anomalies_stud(n=1, mult_var=5, mult_moy=1,seuil=0.99):
    seuil = seuil_student(seuil,mult_var,mult_moy)
    
    anomalies = []
    while len(anomalies)<n:
        valeurs = mult_student(5*n,mult_var,mult_moy)
        khi2 = np.sum(np.multiply(valeurs,valeurs),axis=1)
        indices = [i for i in range(5*n) if khi2[i]> seuil]
        new_anomalies = np.array(valeurs)[indices]
        anomalies.extend(new_anomalies)
    
    anomalies = np.array(anomalies[:n])

    return anomalies


anomalies = anomalies_stud(1000,.05,1,seuil=.99)
print(np.mean(anomalies,axis=0))
_,nb=detection(valeurs=anomalies,seuil=0.99,verbose=True)



# Test inversion matrice Cov
var = np.cov(np.transpose(X_train))
test = np.random.multivariate_normal(np.zeros(X_size),var,10000)
svar = np.linalg.inv(np.linalg.cholesky(var))
np.cov(np.transpose(test))

test2 = np.dot(test,svar.T)
np.cov(np.transpose(test2))

