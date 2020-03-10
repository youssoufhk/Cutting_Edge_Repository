#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 16:05:19 2020
@author: chenzeyu
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KernelDensity

#Reset all the graph 
tf.reset_default_graph()

## Import our raw data
data_close = pd.read_csv('data.csv')

n_time = data_close.shape[0]
n_stocks = data_close.shape[1]


# Data treatment ,fill the NA and calculate the daily reture
for j in range(0,n_stocks):
        data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())

#Data=(data_close)/np.std(data_close)

        
Data = np.log(data_close.values[1:,:]/data_close.values[:-1,:])

X_data =  Data[:,0:5]*100
#X_data =  Data[:,np.random.randint(0,237,5)]*100

#Split the raw data to two part train and test


X_train, X_test = train_test_split(X_data, test_size = 0.35,shuffle=False)
#X_train, X_test = train_test_split(X_data, test_size = 0.35,random_state=42)
#Constants declaration 
Y_size = X_train.shape[0]     #the number of date we will used for one network training
X_size = X_train.shape[1]     #X_size is the number of stock
epochs = 8000                #the number of iteration

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


## Plot les loss
def plot_loss(d_loss_list,g_loss_list):
    plt.subplot(2, 1, 1) 
    plt.plot(d_loss_list, 'yo-') 
    plt.ylabel('d_loss')

    plt.subplot(2, 1, 2)  
    plt.plot(g_loss_list,'r.-')  
    plt.ylabel('g_loss')
    
def generator(Z,nb_neurone=[64,32],reuse=False):
    """ generator structure
    Args:
        Z: The noise that is uniform or Gaussain
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Generator",reuse=reuse):
        h1 = tf.layers.dense(Z,64,activation=tf.nn.leaky_relu)
        #h2 = tf.layers.dense(h1,32,activation=tf.nn.leaky_relu)
        #h3 = tf.layers.dense(h2,16,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,X_size)
    return output


def discriminator(X,nb_neurone=[64,32],reuse=False):
    """ generator structure
    Args:
        X: The real data or generated data 
        nb_neurone : number of neurone of one layer
        reuse: False means create a new variable,True means reuse the existing one 
    """
    with tf.variable_scope("GAN/Discriminator",reuse=reuse):
        h1 = tf.layers.dense(X,64,activation=tf.nn.leaky_relu)
        #h2 = tf.layers.dense(h1,32,activation=tf.nn.leaky_relu)
        #h3 = tf.layers.dense(h2,16,activation=tf.nn.leaky_relu)
        output = tf.layers.dense(h1,1)

    return output

X = tf.placeholder(tf.float32,[None,X_size])
Z = tf.placeholder(tf.float32,[None,X_size])



gen_sample = generator(Z)
real_logits = discriminator(X)
gen_logits = discriminator(gen_sample,reuse=True)

#corr = tf.transpose(tfp.stats.correlation(gen_sample))
#corr_loss = tf.reduce_sum(corr)- tf.reduce_sum(tf.diag_part(corr))


"""
Definition of loss function and the Optimizer with learning rate  0.001, decay 0.95
"""
#dis_loss = min E[-log(D(X))] + E[log(1-D(G(Z)))] := real_loss + gen_loss
#sigmoid_cross_entropy_with_logits(x,z) = z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
r_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits,labels=tf.ones_like(real_logits))
g_loss=tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_logits,labels=tf.zeros_like(gen_logits))
disc_loss = tf.reduce_mean(r_loss + g_loss)


#gen_loss = min E[log(1-D(G(Z)))] =  max  E[log D(G(Z)] = min - E[log(D(G(Z)))]
gen_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=gen_logits,labels=tf.ones_like(gen_logits)))



gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Generator")
disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="GAN/Discriminator")
gen_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.9, momentum=0.0, epsilon=1e-8).minimize(gen_loss,var_list = gen_vars) 
disc_step = tf.train.RMSPropOptimizer(learning_rate=0.001,decay=0.9, momentum=0.0, epsilon=1e-8).minimize(disc_loss,var_list = disc_vars) 



sess = tf.Session()
sess.run(tf.global_variables_initializer())
nd_steps=1#entrainer plus de dis que gen
ng_steps=1
d_loss_list=[]
g_loss_list = []


"""
Training process
"""

X_batch = X_train
with tf.device('/device:GPU:0'):
    for i in range(epochs):
        #Z_batch = sample_noise_uniform(Y_size,X_size)
        Z_batch = sample_noise_Gaus(Y_size,X_size)
        #ind_X = random.sample(range(Y_size),Y_size)
        for _ in range(nd_steps):
            _, dloss = sess.run([disc_step, disc_loss], feed_dict={X: X_batch, Z: Z_batch})
           
        for _ in range(ng_steps):
            _, gloss = sess.run([gen_step, gen_loss], feed_dict={Z: Z_batch})
            
        if i%100==0:    
            print ("Iterations:" ,i,"Discriminator loss: ",dloss,"Generator loss:" , gloss)
            d_loss_list.append(dloss)
            g_loss_list.append(gloss)

 

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
    score = -sum(kde.score_samples(Y))
    return score

       
#Generate data with our generator by feeding Z 
Z_batch = sample_noise_Gaus(Y_size,X_size)
pred=sess.run(gen_sample,feed_dict={Z: Z_batch})
print("The score of predition is :", KDE(X_batch,pred),"The best score is :", KDE(X_batch,X_batch))

#Check if generator cheated discriminator by checking if Prob_real and
#Prob_pred are closed to 0.5
y_real=sess.run(real_logits,feed_dict={X: X_batch})

Prob_real=sess.run(tf.sigmoid(y_real))

y_pred=sess.run(real_logits,feed_dict={X: pred})

Prob_pred=sess.run(tf.sigmoid(y_pred))


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

#plot the loss
plt.figure(num=0, figsize=(7, 5))

plot_loss(d_loss_list,g_loss_list)

plt.figure(num=1, figsize=(7, 5))

D0 = pd.DataFrame(np.transpose((X_batch[:,0],pred[:,0]))) 
D0.plot.density()
plt.xlim((-25, 25))
plt.title('return series of stock 1')

plt.figure(num=2, figsize=(7, 5))

D1 = pd.DataFrame(np.transpose((X_batch[:,1],pred[:,1]))) 
D1.plot.density()
plt.xlim((-25, 25))
plt.title('return series of stock 2')
plt.show()

plt.figure(num=3, figsize=(7, 5))

D2 = pd.DataFrame(np.transpose((X_batch[:,2],pred[:,2]))) 
D2.plot.density()
plt.xlim((-25, 25))
plt.title('return series of stock 3')
plt.show()
"""
plt.figure(num=1, figsize=(7, 5))
plt.ylim((-15, 15))
plt.plot(X_batch[:,0])
plt.plot(pred[:,0])
plt.title('return series of stock 1')

plt.figure(num=2, figsize=(7, 5))
plt.plot(X_batch[:,1])
plt.plot(pred[:,1])
plt.title('return series of stock 2')
plt.ylim((-15, 15))
plt.show()

plt.figure(num=3, figsize=(7, 5))
plt.plot(X_batch[:,2])
plt.plot(pred[:,2])
plt.title('return series of stock 3')
plt.ylim((-15, 15))
plt.show()

plt.figure(num=4, figsize=(7, 5))
plt.plot(X_batch[:,3])
plt.plot(pred[:,3])
plt.title('return series of stock 4')
plt.ylim((-15, 15))
plt.show()

plt.figure(num=5, figsize=(7, 5))
plt.plot(X_batch[:,4])
plt.plot(pred[:,4])
plt.title('return series of stock 5')
plt.ylim((-15, 15))
plt.show()

"""