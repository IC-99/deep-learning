import numpy as np
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras.layers import Input, Concatenate, Conv1D, Activation, Dense, Flatten
from tensorflow.keras.models import Model

# Definizione del modello
def build_model(temporal_features_length, constant_features_length):
    # Input layer per le variabili temporali
    temporal_input = Input(shape=(7, temporal_features_length))
    
    # Input layers per le variabili costanti
    constant_inputs = Input(shape=(constant_features_length,))
    
    # Flatten delle variabili temporali per passarle attraverso il layer Conv1D
    x = Flatten()(temporal_input)
    print(x)
    
    # Concatenazione degli input
    concatenated_inputs = Concatenate()([x, constant_inputs])
    
    # Aggiungi uno o più blocchi Conv1D per elaborare gli input concatenati
    for dilation_rate in [1, 2, 4, 8, 16]:
        x = Conv1D(filters=64, kernel_size=2, padding='causal', dilation_rate=dilation_rate)(concatenated_inputs)
        x = Activation('relu')(x)
    
    # Aggiungi uno o più strati densi per la previsione finale
    x = Dense(32, activation='relu')(x)
    
    # Strato di output
    output = Dense(6)(x)  # Supponendo 6 variabili temporali da prevedere
    
    # Definizione del modello
    model = Model(inputs=[temporal_input, constant_inputs], outputs=output)
    
    return model

# Esempio di dati
# Supponiamo che X_temporal e X_constant siano i tuoi dati temporali e costanti rispettivamente
X_temporal = np.random.randn(100, 7, 6)  # Esempio di dati temporali con 100 esempi, 7 righe e 6 colonne
X_constant = np.random.randn(100, 3)  # Esempio di dati costanti con 100 esempi e 3 colonne
y = np.random.randn(100, 6)  # Esempio di target con 100 esempi e 6 colonne

for index, series in elenco_utenti.iterrows():
    utente = series['user_id']
    if utente in dataset.index:
        utenti.append(dataset.loc[utente])
len(utenti)


def createXY(dataset_parameter, window_size):
    dataX = []
    dataY = []
    for i in range(window_size, len(dataset_parameter)):
        #per ogni feature fa l'append dei precedenti
        dataX.append(dataset_parameter[i - window_size:i, 0:dataset_parameter.shape[1]])
        dataY.append(dataset_parameter[i])
    return np.array(dataX), np.array(dataY)

train_lista = utenti[:n_utenti_train]

trainX_lista = []
trainY_lista = []

for t in train_lista:
    trainX_temp, trainY_temp = createXY(t.to_numpy(), 7)
    if len(trainX_temp.shape) == 3 and len(trainY_temp.shape) == 1:
        trainX_lista.append(trainX_temp)
        trainY_lista.append(trainY_temp)

trainX = np.concatenate(trainX_lista)
trainY = np.concatenate(trainY_lista)

test_lista = utenti[n_utenti_train:]

testX_lista = []
testY_lista = []

for t in test_lista:
    testX_temp, testY_temp = createXY(t.to_numpy(), 7)
    if len(testX_temp.shape) == 3 and len(testY_temp.shape) == 1:
        testX_lista.append(testX_temp)
        testY_lista.append(testY_temp)

testX = np.concatenate(testX_lista)
testY = np.concatenate(testY_lista)



# Costruzione del modello
model = build_model(temporal_features_length=X_temporal.shape[2], constant_features_length=X_constant.shape[1])

# Compilazione del modello
model.compile(optimizer='adam', loss='mse')

# Addestramento del modello
history = model.fit(x=[X_temporal, X_constant], y=y, epochs=10, batch_size=32, validation_split=0.2)

# Stampa delle statistiche di addestramento
print("Loss di addestramento:", history.history['loss'])
print("Val_loss:", history.history['val_loss'])

score_test = model.evaluate(testX, testY, verbose = 0)