import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, Activation, Dense, Flatten
from tensorflow.keras.models import Model

# Definizione del modello
def build_model(input_shape):
    # Input layer per le variabili temporali
    temporal_input = Input(shape=input_shape)
    
    # Aggiungi uno o più blocchi Conv1D per elaborare gli input temporali
    x = temporal_input
    for dilation_rate in [1, 2, 4, 8, 16]:
        x = Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=dilation_rate)(x)
        x = Activation('relu')(x)
    
    # Aggiungi uno o più strati densi per la previsione finale
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)
    
    # Strato di output con solo 6 neuroni
    output = Dense(1)(x)  # Supponendo 6 variabili temporali da prevedere
    
    # Definizione del modello
    model = Model(inputs=temporal_input, outputs=output)
    
    return model
def createXY(dataset_parameter, window_size):
    dataX = []
    dataY = []
    for i in range(window_size, len(dataset_parameter)):
        #per ogni feature fa l'append dei precedenti
        dataX.append(dataset_parameter[i - window_size:i, 0:dataset_parameter.shape[1]])
        dataY.append(dataset_parameter[i][3])
    return np.array(dataX), np.array(dataY)

# creo una lista di dataframe dove ogni elemento è l'insieme delle date per utente
utenti = []
# carico elenco utenti per scorrere tutti gli id
elenco_utenti = pd.read_csv('./userinfo.csv', usecols=['user_id', 'timezone', 'sex', 'age', 'height'])
print(elenco_utenti.head())

dataset = pd.read_csv('./dataset.csv')

dataset.set_index(['user_id','date'], inplace=True)

print(dataset.head())

for index, series in elenco_utenti.iterrows():
    utente = series['user_id']
    if utente in dataset.index:
        utenti.append(dataset.loc[utente])
        
print(len(utenti))

n_utenti_train = (len(utenti) * 80) // 100

train_lista = utenti[:n_utenti_train]

trainX_lista = []
trainY_lista = []

print('---------------------')
counter = 0
for t in train_lista:
    trainX_temp, trainY_temp = createXY(t.to_numpy(), 7)
    try:
        if trainX_temp.shape[0] == trainY_temp.shape[0] and trainX_temp.shape[1] == 7 and trainX_temp.shape[2] == 6:
            trainX_lista.append(trainX_temp)
            trainY_lista.append(trainY_temp)
    except:
        counter += 1

print(counter, 'elementi hanno dato problemi')
trainX = np.concatenate(trainX_lista)
trainY = np.concatenate(trainY_lista)

test_lista = utenti[n_utenti_train:]

testX_lista = []
testY_lista = []


counter = 0
for t in test_lista:
    testX_temp, testY_temp = createXY(t.to_numpy(), 7)
    try:
        if testX_temp.shape[0] == testY_temp.shape[0] and testX_temp.shape[1] == 7 and testX_temp.shape[2] == 6:
            testX_lista.append(testX_temp)
            testY_lista.append(testY_temp)
    except:
        counter += 1
print(counter, 'elementi hanno dato problemi')

testX = np.concatenate(testX_lista)
testY = np.concatenate(testY_lista)

print('testX:', testX[0])
print('testY:', testY[0])

print(trainX.shape, testX.shape, trainY.shape, testY.shape)

# Costruzione del modello
model = build_model((7,6))

# Compilazione del modello
model.compile(optimizer='adam', loss="mse", metrics=['mae'])

# Addestramento del modello
history = model.fit(x=trainX, y=trainY, epochs=20, batch_size=4096, validation_split=0.2)


# Stampa delle statistiche di addestramento
#print("Loss di addestramento:", history.history['loss'])
#print("Val_loss:", history.history['val_loss'])

score_test = model.evaluate(x=testX, y=testY, verbose = 0)

print(score_test)

prediction = model.predict(testX)

np.set_printoptions(suppress=True)
print('prediction:', prediction)

for i in range(100):
    print(testX[i], '->', testY[i], 'ma predetto:', prediction[i])