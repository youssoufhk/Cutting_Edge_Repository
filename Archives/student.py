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

#Reset all the graph
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

###################################
############  STUDENT  ############
###################################
# Student 
NU_COPULA = n_stocks
N_SIM = 10000
var = np.cov(np.transpose(X_train))

gauss_rv = multivariate_normal(cov=var).rvs(N_SIM).transpose()
chi2_rv = chi2(NU_COPULA).rvs(N_SIM)

mult_factor = np.sqrt(NU_COPULA / chi2_rv)

student_rvs = np.multiply(mult_factor, gauss_rv)
## student_rvs = vector R

inv_student_rvs = t.cdf(student_rvs, NU_COPULA).T

T = []
for i in range(N_SIM):
    T.append(t.ppf(inv_student_rvs[i,:],NU_COPULA))


Students = np.array(T)
#Students

#Split the raw data to two part train and test
student_train, student_test = train_test_split(Students, test_size = 0.3,shuffle=False)








####################################
###### DETECTION D'ANOMALIE ########
####################################
# Renvoie le quantile du khi2 équivalent d'observations multidimensionnelles
def khi2_test(valeur,dim=n_stocks):
    khi2 = np.sum(np.multiply(valeur,valeur),axis=1)
    quantile = chi2.cdf(khi2,df=dim)
    return quantile


## On donne en entrée un dataset supposé gaussien, on centre et on réduit.
## On calcule le quantile khi 2 associé pour chaque observation et on compare au seuil
def detection_simple(data_ref, valeurs, seuil=0.99, verbose=False):
    #dim = len(valeurs[0])
    n =  len(valeurs)
    var = np.cov(np.transpose(data_ref))
    svar = np.linalg.inv(np.linalg.cholesky(var))
    mean = np.mean(data_ref,axis=0)
    valeurs = np.dot(valeurs-mean,svar)
    
    quantile = khi2_test(valeurs)
    indices = [i for i in range(n) if quantile[i]>seuil or quantile[i] < 1-seuil]
    nb_anomalies = len(quantile[indices])
    if verbose:
        print("Nombre de quantiles > ",seuil,":",nb_anomalies,"(%.2f%%)"%(100*nb_anomalies/n))

    anomalies =[np.array(valeurs)[indices],np.array(quantile)[indices]]
    return anomalies, nb_anomalies

_,_ = detection_simple(X_train, X_test,verbose=True)

## On donne en entrée un jeu de données. On utilise l'encoder pour récupérer le bruit associé
## On calcule le quantile khi 2 associé à chaque observation et on compare au seuil
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
## On donne en entrée un jeu de données. On modifie une observation d'un stock.
#On fait le test d'anomalies sur l'observation initiale et sur l'observation modifiée.
def test_detection(data, valeur, ligne=0,colonne=0, seuil=0.99, show_res = False, verbose=False):
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


_ = test_detection(X_test,5,show_res=True)


# Test du seuil de détection d'anomalie par colonne sur les observations de X_test
def recherche_seuil(data_ref, colonne=0, ligne=0, seuil=0.99, nb_pas=50, val_max=5):
    val_test = np.linspace(0,val_max,nb_pas)
    detecte = 0
    nb=0

    while (detecte !=1 and nb<nb_pas):
        val = val_test[nb]
        detecte = test_detection(data_ref, val, ligne=ligne, colonne=colonne,seuil=seuil)
        nb+=1
    if detecte==1:
        print("détection d'anomalie pour la valeur ",nb,": %.2f,"%val,"quantile %.2f%%"%(100*norm.cdf(val)))
    else:
        print("Aucune valeur seuil détectée <",val_max,"pour la colonne",colonne)
    return val
    
for col in range(n_stocks):
    recherche_seuil(X_test, col, seuil=0.99, nb_pas=50, val_max=10)
    
    
# Générateur d'anomalie ligne à ligne
## On génère des normales centrées réduites, on garde que les extrêmes  
## (1 observation générée à la fois, cad une ligne de 5 valeurs par ex)
## Puis on ajoute la moyenne et la variance gonflée (* multiplicateur) du jeu de données
def anomalies(data_ref, n=1, multiplicateur=5, seuil=0.99):
    var = multiplicateur*np.cov(np.transpose(data_ref))
    svar = np.linalg.cholesky(var)
    mean = np.mean(data_ref,axis=0)
    anomalies = []
    for i in range(n):
        non_extreme = True
        while non_extreme:
            anomalie = np.random.multivariate_normal(np.zeros(n_stocks),np.eye(n_stocks),1)
            
            quantile = khi2_test(anomalie)
            if quantile > seuil:
                non_extreme = False
        anomalie = np.dot(svar,anomalie[0])+mean
        anomalies.append(anomalie)
    return anomalies


test_ano = anomalies(X_test, 100,5)
_,_=detection(valeurs=test_ano,seuil=0.99,verbose=True)

# Générateur d'anomalies par paquets
## Idem que fct précédente mais on génère plusieurs lignes à la fois
def anomalies_mult(data_ref, n=1, mult_var=5, mult_moy=1,seuil=0.99):
    var = mult_var*np.cov(np.transpose(data_ref))
    mean = mult_moy*np.mean(data_ref,axis=0)
    svar = np.linalg.cholesky(var)   
        
    anomalies = []
    while len(anomalies)<n:
        valeurs = np.random.multivariate_normal(np.zeros(n_stocks),np.eye(n_stocks),5*n)
        quantile = khi2_test(valeurs)
        condition = tuple([quantile > seuil])
        indices = np.array(range(5*n))[condition]
        new_anomalies = np.array(valeurs)[indices]
        anomalies.extend(new_anomalies)
    
    anomalies = np.array(anomalies[:n])
    anomalies = np.dot(anomalies,svar.T)+mean
    return anomalies

    
anomalies = anomalies_mult(X_test, 1000,20,1,seuil=.99)
print(np.mean(anomalies,axis=0))
_,nb=detection(valeurs=anomalies,seuil=0.99,verbose=True)
  

test_ano = anomalies_mult(X_test, 1000,1)
_,_=detection(valeurs=test_ano,seuil=0.99,verbose=True)
_,_=detection_simple(X_test, valeurs=test_ano,seuil=0.99,verbose=True)

# Affichage de la détection d'anomalies
def plot_anomalies(x,y,title='',vol=True):
    plt.plot(x[0],x[1], 'ro-')
    plt.plot(y[0],y[1], 'yo-')
    plt.ylabel('% anomalies détectées')
    plt.ylabel('% anomalies détectées')
    plt.title(title)
    if vol:
        plt.xlabel('Multiplicateur de la vol')
    if not vol:
        plt.xlabel('Multiplicateur de la moyenne')
    plt.grid()
    plt.show()
    
    
# Détection des anomalies multivariées générées
## On génère des anomalies gaussiennes avec vol et/ou mean gonflées
## On regarde cb sont détectées comme anomalies avec méthode BiGAN
def test_anomalies_mult(data_ref, n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_mult(data_ref, n,mult_var,mult_moy,seuil)
    _,nb=detection(valeurs=anomalies,seuil=0.99,verbose=False)
    pct = 100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies,pct

data_ref = X_test

# Multiplication sur la var
pct_anomalies_v = []
for mult in np.linspace(.1,2.5,20):
    _,pct = test_anomalies_mult(data_ref, n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_v.append([mult,pct])
pct_anomalies_v=pd.DataFrame(pct_anomalies_v)

    
    
# Multiplication sur la moyenne    
pct_anomalies_m = []   
for mult in np.linspace(1,45,20):
    _,pct = test_anomalies_mult(data_ref, n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_m.append([mult,pct])    
pct_anomalies_m=pd.DataFrame(pct_anomalies_m)


plt.plot(pct_anomalies_v[0],pct_anomalies_v[1], 'ro-')
plt.ylabel('% anomalies détectées')
plt.xlabel('Multiplicateur de la moyenne')
plt.grid()
plt.show()

# Détection simple (benchmark) des anomalies multivariées générées
## Idem que fct précédente mais on détecte les anomalies en faisant juste un test du khi 2 directement (pas sur les labels)
def test_anomalies_mult_simp(data_ref, n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_mult(data_ref, n,mult_var,mult_moy,seuil)
    _,nb=detection_simple(data_ref, valeurs=anomalies,seuil=0.99,verbose=False)
    pct = 100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies, pct

data_ref = X_test

# Multiplication sur la var
pct_anomalies_v_s = []
for mult in np.linspace(.1,2.5,20):
    _,pct = test_anomalies_mult_simp(data_ref, n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_v_s.append([mult,pct])
pct_anomalies_v_s=pd.DataFrame(pct_anomalies_v_s)
# Multiplication sur la moyenne    
pct_anomalies_m_s = []   
for mult in np.linspace(1,45,20):
    _,pct = test_anomalies_mult_simp(data_ref, n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_m_s.append([mult,pct])    
pct_anomalies_m_s=pd.DataFrame(pct_anomalies_m_s)

plot_anomalies(pct_anomalies_v,pct_anomalies_v_s,"Anomalies Gaussiennes")
plot_anomalies(pct_anomalies_m,pct_anomalies_m_s,vol=False)

# Génère des données de Student multivariée
def mult_student(nb=1,mult_var=1,mult_moy=1,verbose=False):
    var = mult_var*np.cov(np.transpose(X_train))
    mean = mult_moy*np.mean(X_train,axis=0)
    NU_COPULA = n_stocks  
    
    gauss_rv = multivariate_normal(mean=mean,cov=var).rvs(nb).transpose()
    chi2_rv = chi2(NU_COPULA).rvs(nb)
    mult_factor = np.sqrt(NU_COPULA / chi2_rv)
    student_rvs = np.multiply(mult_factor, gauss_rv).T
    if verbose:
        for i in range(n_stocks):
            plt.hist(student_rvs[:,i],density=True)
            plt.show()
    return student_rvs

# quantile historique multi dimension de données Student (statistique d'ordre)
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
_,nb=detection(valeurs=anomalies,seuil=0.99,verbose=True)



# Détection des anomalies de Student multivariées générées
def test_anomalies_stud(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_stud(n,mult_var,mult_moy,seuil)
    _,nb=detection(valeurs=anomalies,seuil=0.99,verbose=False)
    pct=100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies,pct
    
# Multiplication sur la var
pct_anomalies_stud_v = []
for mult in np.linspace(.01,1,20):
    _,pct =  test_anomalies_stud(n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_stud_v.append([mult,pct])
pct_anomalies_stud_v=pd.DataFrame(pct_anomalies_stud_v)
# Multiplication sur la moyenne    
pct_anomalies_stud_m = []
for mult in np.linspace(1,16,20):
    _,pct =  test_anomalies_stud(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_stud_m.append([mult,pct])
pct_anomalies_stud_m=pd.DataFrame(pct_anomalies_stud_m)
    
    
# Détection simple (benchmark) des anomalies de Student multivariées générées
def test_anomalies_stud_simp(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_stud(n,mult_var,mult_moy,seuil)
    _,nb=detection_simple(X_test,valeurs=anomalies,seuil=0.99,verbose=False)
    pct=100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies,pct
    
# Multiplication sur la var
pct_anomalies_stud_v_s = []
for mult in np.linspace(.01,1,20):
    _,pct =  test_anomalies_stud_simp(n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_stud_v_s.append([mult,pct])
pct_anomalies_stud_v_s=pd.DataFrame(pct_anomalies_stud_v_s)
# Multiplication sur la moyenne    
pct_anomalies_stud_m_s = []
for mult in np.linspace(1,16,20):
    _,pct =  test_anomalies_stud_simp(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_stud_m_s.append([mult,pct])
pct_anomalies_stud_m_s=pd.DataFrame(pct_anomalies_stud_m_s)

#Affichage des résultats sur les anomalies Student
plot_anomalies(pct_anomalies_stud_v,pct_anomalies_stud_v_s,"Anomalies Student")
plot_anomalies(pct_anomalies_stud_m,pct_anomalies_stud_m_s,vol=False)
