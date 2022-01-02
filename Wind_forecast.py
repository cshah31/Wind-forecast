# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 2021 1:30 pm. 

@author: cshah
This code is a proprietary of Chinmay Shah (cshah@alaska.edu). The main function of the code is to forecast power generation for wind turbine.
License: GNU-GPL 3.0
"""
# Import required libraries
import time
from math import sqrt
import numpy as np
from numpy.core.shape_base import block
from pandas import read_csv
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as md
import seaborn as sns
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # Uncomment this if you want to run on cpu
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, LeakyReLU, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


## FUnctions for the code
# split a univariate dataset into train/test sets
def split_dataset(data):
	# Split 1 hour data into training and test data
    train, test = data[0:14208,:], data[14208:14616,:]
    return train, test

# convert the historical data into the inputs and outputs of 24 hour time period
def to_supervised(data, lookback=1):
	# flatten data
    X, Y = [], []
    for i in range(len(data)-lookback-1):
        a = data[i:(i+lookback), 0]
        X.append(a)
        Y.append(data[i + lookback, 0])
    return np.array(X), np.array(Y)

# Build the lstm model using the Adam optimizer
def build_model(train, n_input):
    # prepare data
    train_x, train_y = to_supervised(train, n_input)
    #print("i want this",train_x.shape[1])
    train_x = np.reshape(train_x, (train_x.shape[0], 1, train_x.shape[1]))
    # define model
    model = Sequential()
    model.add(Bidirectional(LSTM(256, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]))))
    #model.add(LeakyReLU(alpha=0.3))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu'))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    #model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    #model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.1))
    model.add((Dense(1)))
    #model.add(LeakyReLU(alpha=0.1))
    opt = Adam(learning_rate=0.001)
    model.compile(loss='mse', optimizer=opt)
    return train_x, train_y, model

# Train the LSTM model for Day ahead Wind forecast
def train_model(model, train_x, train_y):
    # define parameters
    verbose, epochs, batch_size = 1, 1500, 48
    callback = [EarlyStopping(monitor="loss", min_delta=0.00001, patience=100, mode='auto', restore_best_weights=True)]
    # fit network
    result = model.fit(train_x, train_y, epochs=epochs, callbacks=callback, batch_size=batch_size, verbose=verbose)
    print(model.summary())
    return model, result

# Once the model is trained, make a forecast for the next 24 hours
def forecast(model, history):
    yhat = model.predict(history, verbose=0)
    # we only want the vector forecast
    yhat = yhat[0]
    return yhat

# evaluate a single model
def evaluate_model(train, test, n_input):
	# Build model
    start = time.time()
    train_x, train_y, model = build_model(train, n_input)
    # Train model
    model, result = train_model(model, train_x, train_y)
    stop = time.time()
    print("The model was trained in {}".format(stop-start), "seconds")
    
    # Make Prediction
    test_x, test_y = to_supervised(train, n_input)
    test_x = np.reshape(test_x, (test_x.shape[0], 1, test_x.shape[1]))
    
    # walk-forward validation over each day
    predictions = list()
    for i in range(len(test_x)):
        # history is a list of daily data
        history = test_x[i]
        history = np.reshape(history, (history.shape[0], 1, history.shape[1]))
        # predict the next hour
        yhat_sequence = forecast(model, history)
        # store the predictions
        predictions.append(yhat_sequence)
        
    predictions = np.array(predictions)

    return test_y, result, predictions

# summarize scores
def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))


## Main part of the code
Wind_dataset = pd.read_csv('\path-to-your-data\your-data.csv', header=0, infer_datetime_format=True)
data = Wind_dataset['Wind (kW)'].values
data = data.reshape(len(data),1)
#########################################################################################################################################################
################### Scaling the data ################################
scale_data = MinMaxScaler(feature_range=(0,1))
#scale_data = StandardScaler()
scale_data = scale_data.fit(data)
normalized_Wind_dataset = scale_data.transform(data)
print(normalized_Wind_dataset.shape)
#########################################################################################################################################################
# Split the dataset into training and test sets
train, test = split_dataset(normalized_Wind_dataset)
print(train.shape)
# evaluate model and get scores
n_input = 48       # best value:144
test_y, result, predictions = evaluate_model(train, test, n_input)

test_y = np.reshape(test_y, (len(test_y), 1))
Actual_Wind_f = scale_data.inverse_transform(test_y)
Predicted_Wind_f = scale_data.inverse_transform(predictions)
print(Actual_Wind_f.shape)
print(Predicted_Wind_f.shape)

print('Test Mean Absolute Error:', mean_absolute_error(Actual_Wind_f[:,0], Predicted_Wind_f[:,0]))
print('Test Root Mean Squared Error:',np.sqrt(mean_squared_error(Actual_Wind_f[:,0], Predicted_Wind_f[:,0])))

# Plot Loss
plt.plot(result.history['loss'])
plt.title('model train loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train'], loc='upper right')

#aa=[x for x in range(24)]
aa = Wind_dataset['Time'].iloc[14208+n_input:14208+n_input+48]
aa = pd.to_datetime(aa, format='%m/%d/%Y %H:%M')
fig, ax = plt.subplots(figsize=(8,4))
plt.plot(aa, Actual_Wind_f[:48], marker='.', label="actual")
plt.plot(aa, Predicted_Wind_f[:48], 'r', label="prediction")
ax.set_xlim(aa.min()-pd.Timedelta(1,'h'),
            aa.max()+pd.Timedelta(1,'h'))
ax.xaxis.set_major_locator(md.HourLocator(interval = 1))
ax.xaxis.set_major_formatter(md.DateFormatter('%m/%d/%Y %H:%M'))
plt.xticks(fontsize=8, rotation=90)
plt.yticks(fontsize=8)
# plt.tick_params(left=False, labelleft=True) #remove ticks
plt.tight_layout()
sns.despine(top=True)
plt.subplots_adjust(left=0.07)
plt.ylabel('Wind Active Power (kW)', size=15)
plt.xlabel('Time step', size=15)
plt.legend(fontsize=15)

plt.show()

Predicted_hourly_load_f = Predicted_Wind_f.ravel()
df = pd.DataFrame(Predicted_hourly_load_f)
df.to_csv('\path-to-output-folder\output.csv')

