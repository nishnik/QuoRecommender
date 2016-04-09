import nltk 
import csv, collections
#from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import enchant
import numpy
from numpy import genfromtxt
import sys
import json, gensim, logging

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
dictionary = enchant.request_dict("en_US")
tags = []

tags_clus = dict()

def clean_ques(query):
	# here query is list of words that are present in the question
	query = query.lower()# converted to lowercase alphabet
	query = tokenizer.tokenize(query) # tokenized
	return ' '.join(query)

with open('clean.json') as data_file:
	data = json.load(data_file)# type(data)=dict

for line, value in data.items():
	for x in value:
		tags.append(x)

tag_count = collections.Counter(tags)	
tags = []
for i, j in tag_count.most_common():
	tags.append(i)

with open('total_tags.txt', 'w') as w:
	for i in range(len(tags)):
		w.write(clean_ques(tags[i]))	