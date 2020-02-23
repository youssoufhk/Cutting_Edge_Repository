#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
from pandas_datareader import data
import numpy as np

import tensorflow as tf
import datetime as dt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.layers import Dense, Dropout, Input
from keras.models import Model,Sequential
from tqdm import tqdm
#from keras.layers.advanced_activations import ReLU
from itertools import product
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from keras.models import model_from_json
import os

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
noise_size = 1

plt.plot(data_close)
plt.xlabel('Time')
plt.ylabel('Closed Value')




#######################
#### GAN algorithm ####
#######################

## generator
def create_generator(nb_couche,nb_neurone):
    generator=Sequential()
    generator.add(Dense(units=nb_neurone,input_dim=noise_size,activation="relu"))
    #generator.add(ReLU(0.2))
    for _ in range(nb_couche):
        generator.add(Dense(units=nb_neurone,activation="relu"))                                                                                        
    generator.add(Dense(units=X_size,activation="tanh"))   
    generator.summary()
    return generator

g=create_generator(1,2)


## discriminator
def create_discriminator(nb_couche,nb_neurone):
    discriminator=Sequential()
    discriminator.add(Dense(units=nb_neurone,input_dim=X_size,activation="relu"))
    discriminator.add(Dropout(0.3))
    for _ in range(nb_couche):
        discriminator.add(Dense(units=nb_neurone,activation="relu"))
        discriminator.add(Dropout(0.3))
    discriminator.add(Dense(units=1, activation='sigmoid'))
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy', optimizer='adam')
    return discriminator

d = create_discriminator(2,1)


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
def training(distrib,epochs, batch_size,nb_couche,nb_neurone,nb_couche2,nb_neurone2):
    # Creating GAN
    generator= create_generator(nb_couche,nb_neurone)
    discriminator= create_discriminator(nb_couche2,nb_neurone2)
    gan = create_gan(discriminator, generator)

    for e in range(1,epochs+1):
        print("Epoch %d" %e)
        for _ in tqdm(range(batch_size)):
            if distrib == 'normal':
                noise= np.random.normal(0,1, [batch_size,noise_size])
            else:
                noise = np.random.rand(batch_size,noise_size)
            generated_indexes = generator.predict(noise)
            image_batch =X_train_r[np.random.randint(low=0,high=X_train_r.shape[0],size=batch_size)]
            X= np.concatenate([image_batch, generated_indexes])
            y_dis=np.zeros(2*batch_size)
            y_dis[:batch_size]=0.9
            discriminator.trainable=True
            discriminator.train_on_batch(X, y_dis)
            noise= np.random.normal(0,1, [batch_size, noise_size])
            y_gen = np.ones(batch_size)
            discriminator.trainable=False
            gan.train_on_batch(noise, y_gen)
            
    if distrib == 'normal':
        noise = np.random.normal(0,1, [batch_size,noise_size])
    else:
        noise = np.random.rand(batch_size,noise_size)
    
    generated = generator.predict(noise)
    score = KDE(X_train_r,generated)
    
    return score



def find_optimal_model(C=3,N=4,epoch=1):
    neurones = [10*x for x in range(1,N+1)]
    scores = []
    
    for i in range(1,C+1):
        for neur in neurones:
            for j in range(1,C+1):
                for neur2 in neurones:
                    nom = "Gén : " + str(i) + "couches " + str(neur) + "neurones " + "Dis : " + str(j) + "couches " + str(neur2) + "neurones"
                    tmp = [nom,training('normal',epoch, X_train_r.shape[0],i,neur,j,neur2)]
                    scores.append(tmp)
    return scores

res=find_optimal_model(1,2,2)                                                                               
pd.DataFrame(res).to_csv('res.csv',index = False)
