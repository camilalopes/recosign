import matplotlib.pyplot as plt
import tensorflow as tf
import warnings
import numpy
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Conv2D, MaxPooling2D, Flatten, Conv1D, MaxPooling1D

from data import x_treino, x_teste, y_treino, y_teste 

x_treino = minmax_scale(x_treino)
x_teste = minmax_scale(x_teste)
x_treino = x_treino.reshape(140, 300, -1)
x_teste = x_teste.reshape(32, 300, -1)


def rnn_model():
    model = Sequential()
    model.add(LSTM(64, input_shape=(300, 1), activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(30, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-4)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    model.fit(x_treino, y_treino, epochs=10, validation_data=(x_teste, y_teste))

def cnn_model():
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape= (x_treino.shape[1:])))
    model.add(Dropout(0.5))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(28, activation='softmax'))
    opt = tf.keras.optimizers.Adam(lr=1e-2, decay=1e-4)
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.fit(x_treino, y_treino, epochs=10, batch_size=32, validation_data=(x_teste, y_teste))
    model.summary()

cnn_model()
