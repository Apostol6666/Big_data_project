from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation
from sklearn.model_selection import train_test_split


from keras.callbacks import LambdaCallback 
from keras.optimizers import RMSprop 
from keras.callbacks import ReduceLROnPlateau 
import random
import sys 
import re

from keras import utils
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

import numpy as np
import matplotlib.pyplot as plt

#Достаем сонеты из вайла с сонетами
#Помещаем в sonnets массив из соннетов, В виде sonnets[ 1sonnet, 2sonnet...]
def get_sonnets():
    sonnet = '';
    sonnets = []
    lines = []

    file_sonnet = open('sonnets.txt','r')

    for line in file_sonnet: 
        regex = re.compile('Sonnet\s+.+\s*')
        if(re.search( 'Sonnet', line)):
            line=''
        lines.append(line)

    i=0;
    while i!=len(lines):
        sonnet = sonnet + lines[i]
        i+=1;
        if i%14==0:
            sonnets.append(sonnet)
            sonnet = ''
    return sonnets

num_words = 10000

sonnets = get_sonnets()
right_sonnets = get_sonnets()

#Преобразуем все слова из сонетов в их числовое представление, используя Tokenizer
tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(sonnets)
tokenizer.word_index
vocabulary = sorted(list(set(tokenizer.word_index))) 

sequences = tokenizer.texts_to_sequences(sonnets)
sequences_r = tokenizer.texts_to_sequences(right_sonnets)

#Создаем горячие вектора X и Y, которые пойдут на вход LSTM
sequences = pad_sequences(sequences, 100)
sequences_r = pad_sequences(sequences_r, 100)

x_train = pad_sequences(sequences, 100)
y_train = pad_sequences(sequences_r, 100)

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1],1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1],1)

X = np.zeros((x_train.shape[0]+1, x_train.shape[1]+1, len(tokenizer.word_index)+1), dtype = np.bool) 
Y = np.zeros((y_train.shape[0]+1, len(tokenizer.word_index)+1), dtype = np.bool) 

for i, sonnet in enumerate(sequences): 
    for t, word in enumerate(sonnet):
        X[i, t, word] = 1        
    Y[i, sequences_r[i]] = 1

#Тестовая выборка, чтобы узнать, насколько хорошо обучена сеть
train_data, x_test, train_label, y_test= train_test_split(X, Y, test_size=0.2, random_state=4)

model = Sequential()
model.add(LSTM(101, input_shape =(101, len(tokenizer.word_index)+1)))
model.add(Dense(len(tokenizer.word_index)+1))
model.add(Activation('softmax'))

#Запускаем сеть
optimizer = RMSprop(lr = 0.01) 
model.compile(optimizer=optimizer, 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

#Запускаем модель
history = model.fit(X, 
                    Y, 
                    epochs=1,
                    batch_size=221,
                    validation_split=0.1)


result = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", result)


model.save("my_model.h5")
