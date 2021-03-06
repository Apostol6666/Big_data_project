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

#Вытаскиваем сонеты в массив sonnets
def get_sonnets(file):
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

def get_words(file):
    with open(file, 'r') as file1:
        text = file1.read() 
    words = text.split()

    for word in words:
        word = word.strip(string.punctuation) 
    return words

sonnets = get_sonnets('sonnets.txt')
words = get_words('sonnets.txt')

tokenizer = Tokenizer(num_words=num_words)
tokenizer.fit_on_texts(sonnets)
tokenizer.word_index
vocabulary = sorted(list(set(tokenizer.word_index))) 

#Создаем два словаря - один переводит цифры в слова, другой - слова в цифры
num_word = dict((i, c) for i, c in enumerate(vocabulary))
word_num = dict((c, i) for i, c in enumerate(vocabulary))

#Функция генерации номера
def sample_index(preds, t = 1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / t
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas) 

#Функция генерации сонета, length - сколько слов нужно предсказать, до 101 слова
#Так как это максимальная длина сонета
#То есть, функцию можно использовать и чтобы просто генерировать случайные сонет со значением 101,
#И чтобы сгенерировать продолжение какого-либо сонета из нашего файла
def gen_sonnet(length, diversity, end_index):

    start_index = random.randint(0, len(vocabulary) - max_length - 1)
    end_index += start_index
    sentence = [] 

    while start_index != end_index:
        sentence.append(words[start_index])
        start_index +=1

    for i in range(length):
            x_pred = np.zeros((1, max_length, len(vocabulary)+1))
            #В массиве sentence у нас находятся элементы, которые уже взяли из words
            #То есть просто случайные слова для начала
            #И наши предсказанные слова
            
            for t, word in enumerate(sentence):
                word = word.strip(string.punctuation) 
                word = word.lower()
                if word in word_num:
                    x_pred[0, t, word_num[word]] = 1.0
            preds = model.predict(x_pred, verbose = 0)[0]
            next_index = sample_index(preds, diversity)
            next_word = num_word[next_index]
            sentence.append(next_word)
    return sentence

#words_get - кол-во слов, кот нужно сгенерировать
words_get = 101
words_put = 101 - words_get
result = gen_sonnet(words_get, 0.2, words_put)

#Функция вывода результата в файл
def get_result(put):

    start = result[0:put]
    end = result[put:101]

    def form_result(r): 
        res = ''
        if r:
            for i in range (17):
                r[i*6] = r[i*6].title()
                line = r[i*6:i*6+6]
                for word in line: 
                    res += word
                    res+= ' '
                res += '\n'
        return res

    form = ''
    if start:
        form += form_result(start)
    form += '\n\n'
    if end:
        form += form_result(end)
    return form

print(get_result(words_put))
file_result = 'result.txt'
f = open(file_result,'w')
f.write(get_result(words_put))
f.close()
