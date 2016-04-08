import nltk 
import csv, collections
from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim.models import Phrases
import enchant
import numpy
from numpy import genfromtxt
import sys
import json

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')

def clean_ques(ques):
	ques = ques.lower()
	ques = tokenizer.tokenize(ques)
	return ques

abc = []
with open('text.json') as data_file:    
    data = json.load(data_file)

ques = list(data.keys())

ques_stream = []
for i in range(len(ques)):
	ques_stream.append(clean_ques(ques[i]))

bigram = Phrases(ques_stream)
trigram = Phrases(bigram[ques_stream])

print (trigram[bigram[clean_ques('This is the New York life in this big city of san francisco and watching game of thrones')]])
#for any new sentence we will be using (trigram[bigram[sent]])
