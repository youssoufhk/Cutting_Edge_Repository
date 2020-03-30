#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Importing libraries
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

# Fixing a seed s.t. the results stay tractable
tf.set_random_seed(2020)
np.random.seed(7)

# Setting the path from which the data is uploaded
default_path = "D:/Documents/Cours/Cutting Edge/"
os.chdir(default_path)

# Reset all the graphs
tf.reset_default_graph()

# Import our raw data
n_stocks = 5 # number of stocks we decided to keep from the data
i = 0        # from which stock should we start
data_close = pd.read_csv('data.csv')
data_close = data_close.iloc[:,i:(i+n_stocks)]

n_time = data_close.shape[0]  # number of points in time

# Data treatment, fill the NA and calculate the daily return
for j in range(0,n_stocks):
    data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())

Data = np.log(data_close.values[1:,:]/data_close.values[:-1,:])

X_data =  Data*100 # 1/np.std(Data) this treatment gives us better values than when we use small numbers

# Split the raw data to two parts train and test

X_train, X_test = train_test_split(X_data, test_size = 0.3,shuffle=False)

#Constants declaration
n_time = X_train.shape[0]     
n_stock = X_train.shape[1]    

epochs = 2000               # number of iterations


#########################################################################################################
# Generating student data 


def generation_student(n_sim=1005,X_train_g=X_train):
    # Student
    NU_COPULA = n_stocks
    N_SIM = n_sim
    var = np.cov(np.transpose(X_train_g))
    
    gauss_rv = multivariate_normal(cov=var).rvs(N_SIM).transpose()
    chi2_rv = chi2(NU_COPULA).rvs(N_SIM)
    
    mult_factor = np.sqrt(NU_COPULA/chi2_rv)
    
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
    return X_train,X_test

student_train, student_test = generation_student(10000,X_train)

n_time_student = student_train.shape[0]     
n_stock_student = student_train.shape[1]     

# In order to use the generated student data, we need to compute the following line
# X_train, X_test, n_time, n_stock = student_train, student_test, n_time_student, n_stock_student

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=n_time, dim=n_stock):
    return np.random.uniform(-1,1,(n,dim))

def sample_noise_Gaus(n=n_time, dim=n_stock):
    var = np.var(X_train)
    mean = np.mean(X_train)
    return np.random.normal(0,1,(n,dim))

def sample_noise_multiGaus(n=n_time):
    var = np.cov(np.transpose(X_train))
    mean = np.mean(X_train,axis=0)
    return np.random.multivariate_normal(mean,var,n)

## Define the structures of the BiGAN

def generator(Z,nb_neurone=[64,32,16],reuse=False):
    """ generator structure
        Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("BiGAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,nb_neurone[0],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,n_stock)
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
    with tf.variable_scope("BiGAN/Discriminator",reuse=reuse):
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
    with tf.variable_scope("BiGAN/Encoder",reuse=reuse):
        h1 = tf.layers.dense(X,nb_neurone[0],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,n_stock)
    return output


## How the BiGAN works

X = tf.placeholder(tf.float32,[None,n_stock])
Z = tf.placeholder(tf.float32,[None,n_stock])

gen_sample = generator(Z)
z_sample = encoder(X)

# The discriminator receives the real data with the encoder's noise (X,E(X)) and the fake data with the random noise (G(Z),Z)
real_output = discriminator(X, z_sample)
fake_output = discriminator(gen_sample, Z,reuse=True)


#Discriminator loss
disc_loss = -tf.reduce_mean(tf.log(real_output+1e-5) + tf.log(1.0-fake_output+1e-5))

#Generator loss
gen_loss = -tf.reduce_mean(tf.log(fake_output+1e-5) + tf.log(1.0-real_output+1e-5))

#Define the Optimizer with learning rate  0.001

gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BiGAN/Generator")

disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BiGAN/Discriminator")

enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BiGAN/Encoder")

gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars+ enc_vars)
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars)


# Training the BiGAN

sess = tf.Session()
sess.run(tf.global_variables_initializer())
nd_steps=2 # train the discriminator twice more because of the double entries
ng_steps=1
d_loss_list=[]
g_loss_list = []

X_batch = X_train

with tf.device('/device:GPU:0'):
    for i in range(epochs):
        Z_batch = sample_noise_Gaus(n_time,n_stock)

        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
        
        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss],  feed_dict={X: X_batch, Z: Z_batch})
        
        if i%100==0:
            print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)
            d_loss_list.append(dloss)
            g_loss_list.append(gloss)



## Plot the respective losses
def plot_loss(d_loss_list,g_loss_list):
    plt.subplot(2, 1, 1)
    plt.plot(d_loss_list, 'yo-')
    plt.ylabel('d_loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(g_loss_list,'r.-')
    plt.ylabel('g_loss')

plt.figure(num=0, figsize=(7, 5))

plot_loss(d_loss_list,g_loss_list)

###########################################################################################
###########################################################################################
# Defining the Kernel Density Evaluation function in order to evaluate if our generated data is alike the real one
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

#Generate data with our generator by feeding the output of the encoder

Z_batch = sess.run(z_sample,feed_dict={X: X_batch})

pred=sess.run(gen_sample,feed_dict={Z: Z_batch})

print("The score of predition is :", KDE(X_batch,pred),"The best score is :", KDE(X_batch,X_batch))

#Check if generator cheated discriminator by checking if Prob_real and Prob_pred are close to 0.5

y_real=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})

Prob_real=sess.run(y_real)

y_pred=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})

Prob_pred=sess.run(y_real)

# Check if the Cov and mean are good (close enough)

np.set_printoptions(suppress=True)

Mean_pred = np.mean(np.transpose(pred),axis=1)
Mean_X = np.mean(np.transpose(X_batch),axis=1)
Cov_pred = np.around(np.cov(np.transpose(pred)), decimals=3)
Cov_X = np.around(np.cov(np.transpose(X_batch)), decimals=3)

# Checking the correlation matrices
Corr_pred = np.around(np.corrcoef(np.transpose(pred)), decimals=3)
Corr_X = np.around(np.corrcoef(np.transpose(X_batch)), decimals=3)

# A plot of the marginals that lets us see how fitted is the fake data to the real

plt.figure(num=1, figsize=(7, 5))
D0 = pd.DataFrame(np.transpose((X_batch[:,0],pred[:,0]))) # 0 represents the index of the stock we want to plot
D0.columns = ['real','fake']
D0.plot.density()
#plt.ylim((-0.05, 0.4))
#plt.xlim((-25, 25))        # some limits in order to see closely the tendancy of the curves
plt.title('return series of stock'+str(0))

##############################################################################################################################
##############################################################################################################################


# Returns khi2 quantile equivalent using multivariate observations
def khi2_test(valeur,dim=n_stock):
    khi2 = np.sum(np.multiply(valeur,valeur),axis=1)
    quantile = chi2.cdf(khi2,df=dim)
    return quantile

# Returns the anomalies detected and their number
# Using the reduce and centred random variable

def detection_simple(valeurs, seuil=0.99, verbose=False):
    
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


# Returns the anomalies detected and their number
# Using the variable as it is

def detection(valeurs, seuil=0.99, verbose=False):
    n =  len(valeurs)
    label = sess.run(z_sample,feed_dict={X: valeurs})
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


# Detection test on a modified value of X_test data set

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

# Simple Detection test on a modified value of X_test data set
def test_detection_simple(valeur, ligne=0,colonne=0, seuil=0.99, show_res = False, verbose=False):
    valeur_ini=[X_test[ligne,:]]
    valeur_mod = deepcopy(valeur_ini)
    valeur_mod[0][colonne] = valeur
    seuil = 0.99
    _,nb_ini = detection_simple(valeur_ini, seuil, verbose)
    _,nb_mod = detection_simple(valeur_mod, seuil, verbose)
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

# Testing the threshold of the anomaly detection using colomuns of X_test

def recherche_seuil(colonne=0, ligne=0, seuil=0.99, nb_pas=20, val_max=10,verbose=False):
    val_test = np.linspace(0,val_max,nb_pas)
    detecte = 0
    nb=0
    
    while (detecte !=1 and nb<nb_pas):
        val = val_test[nb]
        detecte = test_detection(val, ligne=ligne, colonne=colonne,seuil=seuil)
        nb+=1
    if verbose==True:
        if detecte==1:
            print("détection d'anomalie pour la valeur ",nb,"iem : %.2f,"%val,"quantile %.2f%%"%(100*norm.cdf(val)))
        else:
            print("Aucune valeur seuil détectée <",val_max,"pour la colonne",colonne)
    return val

# Testing the threshold of the simple anomaly detection using colomuns of X_test
def recherche_seuil_simple(colonne=0, ligne=0, seuil=0.99, nb_pas=20, val_max=10,verbose=False):
    val_test = np.linspace(0,val_max,nb_pas)
    detecte = 0
    nb=0
    
    while (detecte !=1 and nb<nb_pas):
        val = val_test[nb]
        detecte = test_detection_simple(val, ligne=ligne, colonne=colonne,seuil=seuil)
        nb+=1
    if verbose==True:
        if detecte==1:
            print("détection d'anomalie pour la valeur ",nb,"iem : %.2f,"%val,"quantile %.2f%%"%(100*norm.cdf(val)))
        else:
            print("Aucune valeur seuil détectée <",val_max,"pour la colonne",colonne)
    return val


# Plot the anomaly detection col by col
def plot_anomalies_col(x,y,title):
    plt.figure(figsize=(8, 6))
    plt.plot(x, 'ro-',label="BIGAN")
    plt.plot(y, 'yo-',label="Simple Method")
    plt.legend(loc='upper right')
    plt.ylabel('anomalies détectées')
    plt.xlabel('Index of Stock')
    plt.title(title)
    plt.grid()
    plt.show()

"""
    val=[]
    val_simple=[]
    for col in range(n_stocks):
    seuil=0
    seuil_simple=0
    for line in range(np.shape(X_test)[0]):
    seuil = seuil+recherche_seuil(col,line,seuil=0.99, nb_pas=50, val_max=20)
    seuil_simple = seuil_simple+recherche_seuil_simple(col,line,seuil=0.99, nb_pas=50, val_max=20)
    val.append(seuil/np.shape(X_test)[0])
    val_simple.append(seuil_simple/np.shape(X_test)[0])
    
    plot_anomalies_col( val,val_simple,"Historique")
    """

###########################################################################################################
# Generating anomalies row by row

def anomalies(n=1, multiplicateur=5, seuil=0.99):
    var = multiplicateur*np.cov(np.transpose(X_train))
    svar = np.linalg.cholesky(var)
    mean = np.mean(X_train,axis=0)
    anomalies = []
    for i in range(n):
        non_extreme = True
        while non_extreme:
            anomalie = np.random.multivariate_normal(np.zeros(n_stock),np.eye(n_stock),1)
            
            quantile = khi2_test(anomalie)
            if quantile > seuil:
                non_extreme = False
        anomalie = np.dot(svar,anomalie[0])+mean
        anomalies.append(anomalie)
    return anomalies


test_ano = anomalies(100,5)
_,_=detection(valeurs=test_ano,seuil=0.99,verbose=True)

# Package of anomalies generation
def anomalies_mult(n=1, mult_var=5, mult_moy=1,seuil=0.99):
    var = mult_var*np.cov(np.transpose(X_train))
    mean = mult_moy*np.mean(X_train,axis=0)
    svar = np.linalg.cholesky(var)
    
    anomalies = []
    while len(anomalies)<n:
        valeurs = np.random.multivariate_normal(np.zeros(n_stock),np.eye(n_stock),5*n)
        quantile = khi2_test(valeurs)
        condition = tuple([quantile > seuil])
        indices = np.array(range(5*n))[condition]
        new_anomalies = np.array(valeurs)[indices]
        anomalies.extend(new_anomalies)
    
    anomalies = np.array(anomalies[:n])
    anomalies = np.dot(anomalies,svar.T)+mean
    return anomalies




######################################################################################################################
#Statistic Indicators
#Performance table
######################################################################################################################

def performance_table_Bigan():
    n=np.shape(X_test)[0]
    accuracy=[]
    precision=[]
    recall=[]
    f_score=[]
    for val in np.linspace(.1,3,25):
        anomalies = anomalies_mult(n,mult_var=val,mult_moy=1,seuil=.99)
        _,TN=detection(valeurs=anomalies,seuil=0.99)
        FP = n-TN
        _,FN=detection(valeurs=X_test,seuil=0.99)
        TP= n-FN
        acc=(TP+TN)/(TP+FP+FN+TN)
        accuracy.append(acc)
        pre = TP/(TP+FP)
        precision.append(pre)
        rec = TP/(TP+FN)
        recall.append(rec)
        f_score.append(2*(rec*pre)/(rec+pre))

    plt.figure()
    plt.plot(np.linspace(.1,3,25),accuracy,'ro-',label="accuracy")

    plt.plot(np.linspace(.1,3,25),precision,'yo-',label="precision")
    
    plt.plot(np.linspace(.1,3,25),recall,'go-',label="recall")
    
    plt.plot(np.linspace(.1,3,25),f_score,'bo-',label="f_score")
    plt.legend(loc='lower right')
    plt.title("Performance table Historique-Bigan")

def performance_table_Simple():
    n=np.shape(X_test)[0]
    accuracy=[]
    precision=[]
    recall=[]
    f_score=[]
    for val in np.linspace(.1,5,25):
        anomalies = anomalies_mult(n,mult_var=val,mult_moy=1,seuil=.99)
        _,TN=detection_simple(valeurs=anomalies,seuil=0.99)
        FP = n-TN
        _,FN=detection_simple(valeurs=X_test,seuil=0.99)
        TP= n-FN
        acc=(TP+TN)/(TP+FP+FN+TN)
        accuracy.append(acc)
        pre = TP/(TP+FP)
        precision.append(pre)
        rec = TP/(TP+FN)
        recall.append(rec)
        f_score.append(2*(rec*pre)/(rec+pre))
    plt.figure()
    plt.plot(np.linspace(.1,3,25),accuracy,'ro-',label="accuracy")
    plt.plot(np.linspace(.1,3,25),precision,'yo-',label="precision")
    plt.plot(np.linspace(.1,3,25),recall,'go-',label="recall")
    plt.plot(np.linspace(.1,3,25),f_score,'bo-',label="f_score")
    plt.legend(loc='lower right')
    plt.title("Performance table Historique-Simple")

performance_table_Bigan()
performance_table_Simple()
###################################################################################################
###################################################################################################


# Plot the anomaly detection using both the simple method and the BiGAN
# The purpose is to show that the BiGAN is useful

def plot_anomalies(x,y,title='',vol=True):
    plt.plot(x[0],x[1], 'ro-',label="BIGAN")
    plt.plot(y[0],y[1], 'yo-',label="Simple Method")
    plt.ylabel('% anomalies détectées')
    plt.ylabel('% anomalies détectées')
    plt.legend(loc='lower right')
    plt.title(title)
    if vol:
        plt.xlabel('Multiplicateur de la vol')
    if not vol:
        plt.xlabel('Multiplicateur de la moyenne')
    plt.grid()
    plt.show()

# Detecting the multivariate generated anomalies
def test_anomalies_mult(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_mult(n,mult_var,mult_moy,seuil)
    _,nb=detection(valeurs=anomalies,seuil=0.99,verbose=False)
    pct = 100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies,pct

# Multiplication of the var 
pct_anomalies_v = []
for mult in np.linspace(.1,2.5,20):
    _,pct = test_anomalies_mult(n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_v.append([mult,pct])
pct_anomalies_v=pd.DataFrame(pct_anomalies_v)



# Multiplication of the mean
pct_anomalies_m = []
for mult in np.linspace(1,45,20):
    _,pct = test_anomalies_mult(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_m.append([mult,pct])
pct_anomalies_m=pd.DataFrame(pct_anomalies_m)


plt.plot(pct_anomalies_v[0],pct_anomalies_v[1], 'ro-')
plt.ylabel('% anomalies détectées')
plt.xlabel('Multiplicateur de la moyenne')
plt.grid()
plt.show()


# Benchmark on the multivariate generated anomalies
def test_anomalies_mult_simp(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_mult(n,mult_var,mult_moy,seuil)
    _,nb=detection_simple(valeurs=anomalies,seuil=0.99,verbose=False)
    pct = 100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies, pct

# Multiplication of the var
pct_anomalies_v_s = []
for mult in np.linspace(.1,2.5,20):
    _,pct = test_anomalies_mult_simp(n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_v_s.append([mult,pct])
pct_anomalies_v_s=pd.DataFrame(pct_anomalies_v_s)
# Multiplication of the mean
pct_anomalies_m_s = []
for mult in np.linspace(1,45,20):
    _,pct = test_anomalies_mult_simp(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_m_s.append([mult,pct])
pct_anomalies_m_s=pd.DataFrame(pct_anomalies_m_s)

plot_anomalies(pct_anomalies_v,pct_anomalies_v_s,"Anomalies Gaussiennes")
plot_anomalies(pct_anomalies_m,pct_anomalies_m_s,vol=False)

# Generating Multivariate student t distribution
def mult_student(nb=1,mult_var=1,mult_moy=1,verbose=False):
    var = mult_var*np.cov(np.transpose(X_train))
    mean = mult_moy*np.mean(X_train,axis=0)
    NU_COPULA = n_stock
    
    gauss_rv = multivariate_normal(mean=mean,cov=var).rvs(nb).transpose()
    chi2_rv = chi2(NU_COPULA).rvs(nb)
    mult_factor = np.sqrt(NU_COPULA / chi2_rv)
    student_rvs = np.multiply(mult_factor, gauss_rv).T
    if verbose:
        for i in range(n_stock):
            plt.hist(student_rvs[:,i],density=True)
            plt.show()
    return student_rvs

# Historic quantile multi dimension
def quant_hist(x,alpha,verbose=False):
    khi2 = np.sum(np.multiply(x,x),axis=1)
    khi2.sort()
    quantile = khi2[int(alpha*len(khi2))]
    if verbose:
        print("quantile pour alpha=%f"%alpha,":",quantile)
    return quantile

# Threshold alpha for the multivariate student t
def seuil_student(alpha=0.99,mult_var=1,mult_moy=1,rep=100,nb=100000):
    seuil=0
    for i in range(rep):
        valeurs = mult_student(nb,mult_var,mult_moy,verbose=False)
        seuil += quant_hist(valeurs,.99,False)
    seuil=seuil/rep

return seuil

# Generating student anomalies by packages
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


# Detecting the generated multivariate student anomalies
def test_anomalies_stud(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_stud(n,mult_var,mult_moy,seuil)
    _,nb=detection(valeurs=anomalies,seuil=0.99,verbose=False)
    pct=100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies,pct

# Multiplication of the var
pct_anomalies_stud_v = []
for mult in np.linspace(.01,1,20):
    _,pct =  test_anomalies_stud(n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_stud_v.append([mult,pct])
pct_anomalies_stud_v=pd.DataFrame(pct_anomalies_stud_v)

# Multiplication of the mean
pct_anomalies_stud_m = []
for mult in np.linspace(1,16,20):
    _,pct =  test_anomalies_stud(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_stud_m.append([mult,pct])
pct_anomalies_stud_m=pd.DataFrame(pct_anomalies_stud_m)


# Benchmark on the multivariate student t generated anomalies
def test_anomalies_stud_simp(n=1, mult_var=5, mult_moy=1, seuil=0.99,verbose=True):
    anomalies = anomalies_stud(n,mult_var,mult_moy,seuil)
    _,nb=detection_simple(valeurs=anomalies,seuil=0.99,verbose=False)
    pct=100*nb/n
    if verbose:
        print("Seuil: ",seuil,"mult var: %.2f"%mult_var,"mult moy: %.2f"%mult_moy,"Nb quantiles >:",nb,"(%.2f%%)"%(pct))
    return anomalies,pct

# Multiplication of the var
pct_anomalies_stud_v_s = []
for mult in np.linspace(.01,1,20):
    _,pct =  test_anomalies_stud_simp(n=10000,mult_var=mult,seuil=0.99)
    pct_anomalies_stud_v_s.append([mult,pct])
pct_anomalies_stud_v_s=pd.DataFrame(pct_anomalies_stud_v_s)
# Multiplication of the mean
pct_anomalies_stud_m_s = []
for mult in np.linspace(1,16,20):
    _,pct =  test_anomalies_stud_simp(n=10000,mult_var=.1,mult_moy=mult,seuil=0.99)
    pct_anomalies_stud_m_s.append([mult,pct])
pct_anomalies_stud_m_s=pd.DataFrame(pct_anomalies_stud_m_s)

# Plot the different results with student anomalies
plot_anomalies(pct_anomalies_stud_v,pct_anomalies_stud_v_s,"Anomalies Student")
plot_anomalies(pct_anomalies_stud_m,pct_anomalies_stud_m_s,vol=False)

# Cov matrix
var = np.cov(np.transpose(X_train))
test = np.random.multivariate_normal(np.zeros(n_stock),var,10000)
svar = np.linalg.inv(np.linalg.cholesky(var))
np.cov(np.transpose(test))

test2 = np.dot(test,svar.T)
np.cov(np.transpose(test2))

