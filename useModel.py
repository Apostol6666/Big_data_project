from keras.models import load_model
from pymongo import MongoClient 
import numpy as np
from keras.preprocessing.text import Tokenizer
import random 
import string
import re

model = load_model('my_model.h5')

num_words = 10000
max_length = 101

def get_sonnets(file):
    sonnet = '';
    sonnets = []
    lines = []

    file_sonnet = open('sonnets.txt','r')

    for line in file_sonnet: 
        regex = re.compile('Sonnet\s+.+\s*')
        if(re.search( 'Sonnet', line)):
            print(line)
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
sonnets = get_sonnets('sonnets.txt')

with open('sonnets.txt', 'r') as file:
    text = file.read() 

words = text.split()
for word in words:
    word = word.strip(string.punctuation) 

vocabulary = sorted(list(words)) 

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(sonnets)
tokenizer.word_index
vocabulary2 = sorted(list(set(tokenizer.word_index))) 
print(len(vocabulary2))
print(vocabulary2)

indices_to_char = dict((i, c) for i, c in enumerate(vocabulary2))

char_to_indices = dict((c, i) for i, c in enumerate(vocabulary2))
print(char_to_indices)
i=0
while i!=5:
    ch = indices_to_char[i]
    i+=1

def sample_index(preds, temperature = 1.0):

    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)

    return np.argmax(probas) 

def generate_text(length, diversity):

    # Получить случайный начальный текст
    start_index = random.randint(0, len(vocabulary2) - max_length - 1)
    start_index1 = random.randint(0, len(sonnets))
    generated = ''
    end_index = start_index
    sentence = ''
    sen = words[start_index: start_index + 100] 
    sentence = []
    #while end_index!=start_index + max_length:
    sentence.append(words[end_index])
    generated += words[end_index]
    generated += '  '
    end_index +=1

    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary2)+1))
            #print('sentence = ', sentence)
            #print('sonnet =', sonnets[start_index1])
            for t, word in enumerate(sentence):
                #print(t,' ; ', ' ; ' ,word)
                #c = str(sonnets[start_index1]
                #print('word = ', word)
                word = word.strip(string.punctuation) 
                word = word.lower()
                if word in char_to_indices:
                    x_pred[0, t, char_to_indices[word]] = 1.0
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_char = indices_to_char[next_index]
            #print('next_char  =  ', next_char, '  ||||   ')
            generated += next_char
            generated += '  ';
            sentence.append(next_char)
            #sentence = sentence[1:] + next_char
    return generated

print(generate_text(101, 0.2))
