import nltk 
import csv, collections
from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
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
    for line in data:
    	abc.append(json.loads(line))

ques_stream = []
for i in range(len(abc)):
	ques_stream.append(clean_ques(abc[i]))

bigram = Phrases(ques_stream)
trigram = Phrases(bigram[ques_stream])

#for any new sentence we will be using (trigram[bigram[sent]])
