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
	ques = ques.replace('\n', ' ')
	''.join(ch for ch in ques if (ch.isalnum() and not ch.isdigit()))
	ques = tokenizer.tokenize(ques)
	return ques

abc = []
with open('clean.json') as data_file:    
    data = json.load(data_file)

ques = list(data.keys())

ques_stream = []
for i in range(len(ques)):
	ques_stream.append(clean_ques(ques[i]))

bigram = Phrases(ques_stream)
trigram = Phrases(bigram[ques_stream])


for key, value in data.items():
	st = trigram[bigram[clean_ques(key)]]
	s = ' '.join(st)
    d[trigram[bigram[clean_ques(key)]]] = value


print (trigram[bigram[clean_ques('What is the e commerce traffic like from Palo Alto to San Francisco at 3PM on an average weekday?')]])
#for any new sentence we will be using (trigram[bigram[sent]])
