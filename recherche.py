#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pandas_datareader import data
import numpy as np

import datetime as dt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from tqdm import tqdm
#from keras.layers.advanced_activations import ReLU

from sklearn.neighbors import KernelDensity

#import os

#default_path = "D:/Documents/Cours/Cutting Hedge/"
#os.chdir(default_path)

###############################
##### Import des donnees ######
###############################
#Liste d'indices
tickers = ["SPXL","^DJI","^IXIC","FEZ","^FCHI","DAX","^DJI"]
#Liste de sous jacents
stocks = ["MMM","ABT","ABBV","ABMD","ACN","ATVI","ADBE","AMD","AAP","AES","AMG","AFL","A","APD","AKAM","ALK","ALB","ARE","ALXN","ALGN","ALLE","AGN","ADS","LNT","ALL","GOOGL","GOOG","MO","AMZN","AMCR","AEE","AAL","AEP","AXP","AIG","AMT","AWK","AMP","ABC","AME","AMGN","APH","ADI","ANSS","ANTM","AON","AOS","APA","AIV","AAPL","AMAT","APTV","ADM","ARNC","ANET","AJG","AIZ","ATO","T","ADSK","ADP","AZO","AVB","AVY","BKR","BLL","BAC","BK","BAX","BDX","BBY","BIIB","BLK","BA","BKNG","BWA","BXP","BSX","BMY","AVGO","BR","CHRW","COG","CDNS","CPB","COF","CPRI","CAH","KMX","CCL","CAT","CBOE","CBRE","CDW","CE","CNC","CNP","CTL","CERN","CF","SCHW","CHTR","CVX","CMG","CB","CHD","CI","XEC","CINF","CTAS","CSCO","C","CFG","CTXS","CLX","CME","CMS","KO","CTSH","CL","CMCSA","CMA","CAG","CXO","COP","ED","STZ","COO","CPRT","GLW","COST","COTY","CCI","CSX","CMI","CVS","DHI","DHR","DRI","DVA","DE","DAL","XRAY","DVN","FANG","DLR","DFS","DISCA","DISCK","DISH","DG","DLTR","D","DOV","DTE","DUK","DRE","DD","DXC","EMN","ETN","EBAY","ECL","EIX","EW","EA","EMR","ETR","EOG","EFX","EQIX","EQR","ESS","EL","EVRG","ES","RE","EXC","EXPE","EXPD","EXR","XOM","FFIV","FB","FAST","FRT","FDX","FIS","FITB","FE","FRC","FISV","FLT","FLIR","FLS","FMC","F","FTNT","FTV","FBHS","BEN","FCX","GPS","GRMN","IT","GD","GE","GIS","GM","GPC","GILD","GL","GPN","GS","GWW","HRB","HAL","HBI","HOG","HIG","HAS","HCA","PEAK","HP","HSIC","HSY","HES","HPE","HLT","HFC","HOLX","HD","HON","HRL","HST","HPQ","HUM"]
start_date = dt.datetime(2015, 1, 1)
end_date = dt.datetime(2018, 12, 31)
#panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)['Close']
#stocks_data = data.DataReader(stocks,'yahoo',start_date,end_date)['Close']


#panel_data = data.DataReader(tickers, 'yahoo', start_date, end_date)['Close']
#stocks_data = data.DataReader(stocks,'yahoo',start_date,end_date)['Close']
#stocks_data.to_csv('data.csv',index = False)
#data_close = stocks_data
#del stocks_data

data_close = pd.read_csv('data.csv')

n_time = data_close.shape[0]
n_stocks = data_close.shape[1]

## Inference par la moyenne par colomne
for j in range(0,n_stocks):
        data_close.iloc[:,j]=data_close.iloc[:,j].fillna(data_close.iloc[:,j].mean())


## Creation du dataframe des rendements
rendements = np.log(data_close.values[1:,:]/data_close.values[:-1,:])
#rendements = np.transpose(rendements)

X_train_r, X_test_r = train_test_split(rendements, test_size = 0.33, random_state = 42)
#X_train_r = np.transpose(X_train_r)

X_size = X_train_r.shape[1]
noise_size = 128

plt.plot(data_close)
plt.xlabel('Time')
plt.ylabel('Closed Value')




#######################
#### GAN algorithm ####
#######################

## generator
def create_generator(nb_couche,nb_neurone):
    generator=Sequential()
    generator.add(Dense(units=256,input_dim=noise_size,activation="relu"))
    #generator.add(ReLU(0.2))
    for i in nb_neurone:
        generator.add(Dense(units=i))                                                                                        
        #generator.add(ReLU(0.2))
    generator.add(Dense(units=X_size, activation='tanh'))                                                                                                                                                                                     generator.compile(loss='binary_crossentropy', optimizer='adam')
    return generator

g=create_generator(1, [1])


## discriminator
def create_discriminator(nb_couche, nb_neurone):
    discriminator=Sequential()
    discriminator.add(Dense(units=1024,input_dim=X_size,activation="relu"))
    #discriminator.add(ReLU(6.2))
    discriminator.add(Dropout(0.3))
    for i in nb_neurone:
        discriminator.add(Dense(units=i,activation="relu"))
        #discriminator.add(ReLU(0.2))
        discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=256,activation="relu"))
    #discriminator.add(ReLU(0.2))
    discriminator.add(Dense(units=1, activation='sigmoid'))

    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

d =create_discriminator(1, [1])


def create_gan(discriminator, generator):
    discriminator.trainable=False
    gan_input = Input(shape=(noise_size,))
    x = generator(gan_input)
    gan_output= discriminator(x)
    gan= Model(inputs=gan_input, outputs=gan_output)
    gan.compile(loss='binary_crossentropy', optimizer='adam')
    return gan

gan = create_gan(d,g)

## Score du KDE
def KDE(x,y):
    kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(x)
    score = sum(kde.score_samples(y[:,]))
    return score
KDE(X_train_r,X_test_r)


## Train our GAN model
def training(distrib,epochs, batch_size,nb_neurone,nb_couche):
    # Creating GAN
    score = 0

    generator= create_generator(nb_couche,nb_neurone)
    discriminator= create_discriminator(nb_couche,nb_neurone)
    gan = create_gan(discriminator, generator)

    for e in range(1,epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):

            #generate  random noise as an input  to  initialize the  generator
            if distrib == 'normal':
                noise= np.random.normal(0,1, [batch_size,noise_size])
            else:
                noise = np.random.rand(batch_size,noise_size)

            #noise= np.random.normal(0,1, [batch_size,X_size])

            # Generate fake indexes from noised input
            generated_indexes = generator.predict(noise)

            # Get a random set of  real indexes
            image_batch =X_train_r[np.random.randint(low=0,high=X_train_r.shape[0],size=batch_size)]

            #Construct different batches of  real and fake data
            X= np.concatenate([image_batch, generated_indexes])

            # Labels for generated and real data
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9

            #Pre train discriminator on  fake and real data  before starting the gan.
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)

            #Tricking the noised input of the Generator as real data
            noise= np.random.normal(0,1, [batch_size, noise_size])
            y_gen = np.ones(batch_size)

            # During the training of gan,
            # the weights of discriminator should be fixed.
            #We can enforce that by setting the trainable flag
            discriminator.trainable=False

            #training  the GAN by alternating the training of the Discriminator
            #and training the chained GAN model with Discriminators weights freezed.
            gan.train_on_batch(noise, y_gen)
    if distrib == 'normal':
        noise= np.random.normal(0,1, [batch_size,noise_size])
    else:
        noise = np.random.rand(batch_size,noise_size)
    generated = generator.predict(noise)
    score = KDE(X_train_r,generated)

    return score

training('normal',1, 128,[2],1)

def find_optimal_model(C=3,N=4,epoch=1):
    nb_couche = np.array(range(1,C+1))
    nb_neurones = [10*x for x in range(1,N+1)]

    scores = []

    for i in nb_couche:
        for j in np.array(range(0,int(len(nb_neurones)-i+1))):
            nom = ""
            for k in nb_neurones[j:int(j+i)] :
                nom = nom + " C"+str(k)
            tmp = [nom,training('normal',epoch, 128,nb_neurones[j:int(j+i)],i)]
            scores.append(tmp)
    return scores


score2 = find_optimal_model(3,3,10)
print(score2)
pd.DataFrame(score2).to_csv("score.csv")

                                                                                      
