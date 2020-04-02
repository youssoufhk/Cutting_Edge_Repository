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

#Reset all the graph

## Import our raw data
data_close = pd.read_csv('data.csv')

n_time = data_close.shape[0]
n_stocks = data_close.shape[1]


# Data treatment ,fill the NA and calculate the daily reture
for j in range(0,n_stocks):
    data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())

#Data=(data_close)/np.std(data_close)


Data = np.log(data_close.values[1:,:]/data_close.values[:-1,:])

X_data =  Data[:,0:5]*100 # 1/np.std(Data)
#X_data =  Data[:,np.random.randint(0,237,5)]*100
np.random.seed(7)
tf.set_random_seed(1234)
#Split the raw data to two part train and test


X_train, X_test = train_test_split(X_data, test_size = 0.35,shuffle=False,random_state=42)
#X_train, X_test = train_test_split(X_data, test_size = 0.35,random_state=42)
#Constants declaration
Y_size = X_train.shape[0]     #the number of date we will used for one network training
X_size = X_train.shape[1]     #X_size is the number of stock
epochs = 5000
#the number of iteration

##############################################
##############################################

## Generate noise
def sample_noise_uniform(n=Y_size, dim=X_size):
    return np.random.uniform(-1,1,(n,dim))

def sample_noise_Gaus(n=Y_size, dim=X_size):
    var = np.var(X_train)
    mean = np.mean(X_train)
    return np.random.normal(mean,var,(n,dim))

def sample_noise_multiGaus(n=Y_size):
    return np.random.multivariate_normal(np.mean(X_train,axis=0),np.cov(np.transpose(X_train)),n)


def generator(noise ,nb_neurone=[10],couches=1,reuse=False):
    """ generator structure
        Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("BIGAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(noise, units=nb_neurone[0], activation=tf.nn.leaky_relu)
        for i in range(couches-1):
            h1 = tf.layers.dense(h1, units=nb_neurone[i+1])
        output = tf.layers.dense(h1, X_size)
    return output # a data set which looks like the real one


def discriminator(X,Z,nb_neurone=[10],couches=1,reuse=False):
    """ discriminator structure
        Args:
        X,Z: The real data X with the noise generated by encoder E(X) or generated data G(Z) with a random noise Z
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    
    with tf.variable_scope("BIGAN/Discriminator",reuse=reuse):
        h = []
        input = tf.concat((X,Z),1)
        h.append(str(0))
        h[0] = tf.layers.dense(input,nb_neurone[0],activation=tf.nn.leaky_relu)
        for i in range(1,couches):
            h.append(str(i))
            h[i] = tf.layers.dense(h[i-1],nb_neurone[i],activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h[couches-1],1, activation=tf.nn.sigmoid)
    return output
# a probability to belong to the real data

def encoder(X,nb_neurone=[10],couches=1,reuse=False):
    """ encoder structure
        Args:
        X: The real data
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one
        """
    with tf.variable_scope("BIGAN/Encoder",reuse=reuse):
        h1 = tf.layers.dense(X, units=nb_neurone[0], activation=tf.nn.leaky_relu)
        for i in range(couches-1):
            h1 = tf.layers.dense(h1, units=nb_neurone[i+1])
        output = tf.layers.dense(h1,X_size)
    return output # a noise generated from the real data

## Plot les loss
def plot_loss(d_loss_list,g_loss_list):
    plt.subplot(2, 1, 1)
    plt.plot(d_loss_list, 'yo-')
    plt.ylabel('d_loss')
    
    plt.subplot(2, 1, 2)
    plt.plot(g_loss_list,'r.-')
    plt.ylabel('g_loss')




def KDE(X,Y):
    """ Evaluation function
        Args:
        X: The real data
        Y : The generated data
        Return:
        score: the score of generated data, the higer score is, the better data is
        """
    h = np.std(X)*(4/3/len(X))**(1/5) #0.5
    kde = KernelDensity(kernel='gaussian', bandwidth=h).fit(X)
    score = -sum(kde.score_samples(Y))
    return score

tf.set_random_seed(1234)
np.random.seed(7)

def score(neurones,epoques=2000,ng=1,nd=2,lr=0.001,verbose=False,show_train=False,fixed=False):
    scores = []
    X_batch = X_train
    for nb_neur in neurones:
        couches = len(nb_neur)
        tf.reset_default_graph()
        if fixed:
            tf.set_random_seed(1234)
            np.random.seed(7)
        X = tf.placeholder(tf.float32,[None,X_size])
        Z = tf.placeholder(tf.float32,[None,X_size])
        
        
        gen_sample = generator(Z,nb_neurone=nb_neur,couches=couches)
        z_sample = encoder(X,nb_neurone=nb_neur,couches=couches)
        
        real_output = discriminator(X, z_sample,nb_neurone=nb_neur,couches=couches)
        fake_output = discriminator(gen_sample, Z,nb_neurone=nb_neur,couches=couches,reuse=True)
        
        
        #Discriminator loss
        disc_loss = -tf.reduce_mean(tf.log(real_output+1e-5) + tf.log(1.0-fake_output+1e-5))
        
        #Generator loss
        gen_loss = -tf.reduce_mean(tf.log(fake_output+1e-5) + tf.log(1.0-real_output+1e-5))
        
        
        #Define the Optimizer with learning rate  0.001
        
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BIGAN/Generator")
        
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BIGAN/Discriminator")
        
        enc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="BIGAN/Encoder")
        
        gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(gen_loss,var_list = gen_vars+ enc_vars)
        disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001).minimize(disc_loss,var_list = disc_vars)
        
        
        
        
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        d_loss_list=[]
        g_loss_list = []
        
        
        with tf.device('/device:GPU:0'):
            for i in range(epoques):
                Z_batch = sample_noise_Gaus(Y_size,X_size)
                #ind_X = random.sample(range(Y_size),Y_size)
                for _ in range(nd):
                    _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
                
                for _ in range(ng):
                    _, gloss = sess.run([gen_step, gen_loss],  feed_dict={X: X_batch, Z: Z_batch})
                
                if i%100==0:
                    d_loss_list.append(dloss)
                    g_loss_list.append(gloss)
                    if show_train:
                        print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)

        Z_test = sample_noise_Gaus(np.shape(X_test)[0],X_size)
        pred_test=sess.run(gen_sample,feed_dict={Z: Z_test})
        score = KDE(pred_test,X_test)
        print("Score pour C:",couches," N:",nb_neur,", score:",score)
        nom = "C"+str(couches)+"N"+str(nb_neur)
        tmp = [nom,score]
        scores.append(tmp)
        
        
        # Affichage complémentaire
        if verbose:
            # Selection d'un batch de données observées
            index = random.sample(range(Y_size),Y_size)
            X_batch = X_train[index].reshape(Y_size,X_size)
            Z_batch = sample_noise_Gaus(Y_size,X_size)
            pred=sess.run(gen_sample,feed_dict={Z: Z_batch})
            # Vérification du discriminateur
            prob_observe = sess.run(real_output,feed_dict={X: X_batch})
            prob_genere = sess.run(real_output,feed_dict={X: pred})
            
            print("moyenne Discriminateur sur données observées :",np.mean(prob_observe))
            print("moyenne Discriminateur sur données générées :",np.mean(prob_genere))
            
            # Comparaison des distributions observées et estimées
            
            Mean_pred = np.mean(np.transpose(pred),axis=1)
            Mean_X = np.mean(np.transpose(X_batch),axis=1)
            
            print("vrai moyenne:",Mean_X,"\nmoyenne estimée:",Mean_pred)
            Cov_pred = np.cov(np.transpose(pred))
            Cov_X = np.cov(np.transpose(X_batch))
            
            print("cov predite:",Cov_pred)
            print("cov observée:",Cov_X)
            
            plt.hist(pred,density=1,edgecolor='blue',bins=30)
            plt.hist(X_batch,density=1,edgecolor='red',bins=30)
            plt.show()
            
            pd.DataFrame(np.transpose((X_batch[:,0],pred[:,0])),columns = ['real','fake']).plot.density()
    return scores


modeles = [[10],[10,10],[10,10,10],[10,10,10,10],[10,10,10,10,10],
           [20],[20,20],[20,20,20],[20,20,20,20],[20,20,20,20,20],
           [30],[30,30],[30,30,30],[30,30,30,30],[30,30,30,30,30],
           [40],[40,40],[40,40,40],[40,40,40,40],[40,40,40,40,40],
           [50],[50,50],[50,50,50],[50,50,50,50],[50,50,50,50,50],[64],[64,32]]
# fixed : entrainement sensible aux données utilisées/à la graine
scores = score(modeles[0:2],epoques=20,ng=1,nd=1,lr=0.001,verbose=False,show_train=False,fixed=True)
#pd.DataFrame(scores).to_csv('scores.csv',index = False)


#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5
"""
    y_real=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})
    
    Prob_real=sess.run(y_real)
    
    y_pred=sess.run(real_output,feed_dict={X: X_batch,Z:Z_batch})
    
    Prob_pred=sess.run(y_real)
    
    
    #plot the loss
    plt.figure(num=0, figsize=(7, 5))
    
    plot_loss(d_loss_list,g_loss_list)
    
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
