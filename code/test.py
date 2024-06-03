import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, Add, Activation, Lambda
from tensorflow.keras.optimizers import Adam

x = [[1,2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14],[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]
y = [15,15,15,15,15]

# Converti i dati in array numpy
x = np.array(x)
y = np.array(y)

# Definisci i parametri del modello
num_filters = 32
filter_size = 2
dilation_rates = [1, 2, 4, 8]

# Input
inputs = Input(shape=(14, 1))

# Strato di convoluzione causale iniziale
conv_out = Conv1D(filters=num_filters, kernel_size=filter_size, padding='causal', dilation_rate=1)(inputs)

# Strati convoluzionali dilatati
for dilation_rate in dilation_rates:
    conv = Conv1D(filters=num_filters, kernel_size=filter_size, padding='causal', dilation_rate=dilation_rate)(conv_out)
    conv = Activation('relu')(conv)
    conv_out = Add()([conv_out, conv])

# Strato di output finale
outputs = Conv1D(filters=1, kernel_size=1, activation='linear')(conv_out)
outputs = Lambda(lambda x: x[:, -1, :])(outputs)

# Costruisci il modello
model = Model(inputs=inputs, outputs=outputs)

# Compila il modello
model.compile(optimizer=Adam(), loss='mse')

# Reshape dei dati per il modello
x = x[..., np.newaxis]  # aggiungi una dimensione per il canale

# Addestramento del modello
model.fit(x, y, epochs=50, batch_size=32, validation_split=0.2)

res = model.predict(np.array([[1,2,3,4,5,6,7,8,9,10,11,12,13,14]]))

print(res)
