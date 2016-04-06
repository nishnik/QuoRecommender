import nltk 
import csv, collections
from nltk import word_tokenize, pos_tag, sent_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import enchant
import numpy as np
from numpy import genfromtxt
import sys

tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
dictionary = enchant.request_dict("en_US")
stemmer = SnowballStemmer("english")

def clean_ques(ques):
	ques = ques.lower()
	ques = tokenizer.tokenize(ques)
	for i in range(len(ques)):
		if not enchant.dict_exists(ques[i]):
 			ques[i] = dictionary.suggest(ques[i])[0]
	#ques = [q for q in ques if q not in stop]
	ques = [stemmer.stem(q) for q in ques if q not in stop]
	return ques