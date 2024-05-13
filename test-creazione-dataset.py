import numpy as np
import pandas as pd


x_target = np.random.randn(100, 7, 6)  # Esempio di dati temporali con 100 esempi, 7 righe e 6 colonne
y_target = np.random.randn(100, 1, 6)  # Esempio di target con 100 esempi e 6 colonne

def createXY(dataset_parameter, window_size):
    dataX = []
    dataY = []
    for i in range(window_size, len(dataset_parameter)):
        #per ogni feature fa l'append dei precedenti
        dataX.append(dataset_parameter[i - window_size:i, 0:dataset_parameter.shape[1]])
        dataY.append(dataset_parameter[i])
    return np.array(dataX), np.array(dataY)

# creo una lista di dataframe dove ogni elemento è l'insieme delle date per utente
utenti = []
# carico elenco utenti per scorrere tutti gli id
elenco_utenti = pd.read_csv('./userinfo.csv', usecols=['user_id', 'timezone', 'sex', 'age', 'height'], nrows=10000)
print(elenco_utenti.head())

dataset = pd.read_csv('./dataset.csv', nrows=10000)

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
        if trainX_temp.shape[0] == trainY_temp.shape[0] and trainX_temp.shape[1] == 7 and trainX_temp.shape[2] == 6 and trainY_temp.shape[1] == 6:
            trainX_lista.append(trainX_temp)
            trainY_lista.append(trainY_temp)
    except:
        counter += 1

print(counter, 'elementi hanno dato problemi')
trainX = np.concatenate(trainX_lista)
trainY = np.concatenate(trainY_lista)

print('trainX:', trainX[0])
print('trainY:', trainY[0])

test_lista = utenti[n_utenti_train:]

testX_lista = []
testY_lista = []


counter = 0
for t in test_lista:
    testX_temp, testY_temp = createXY(t.to_numpy(), 7)
    try:
        if testX_temp.shape[0] == testY_temp.shape[0] and testX_temp.shape[1] == 7 and testX_temp.shape[2] == 6 and testY_temp.shape[1] == 6:
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