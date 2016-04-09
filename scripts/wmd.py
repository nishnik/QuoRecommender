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


model = gensim.models.Word2Vec.load('/home/adarsh/wordmodel')
with open('clean.json') as data_file:
	data=json.load(data_file)# type(data)=dict
ques=list(data.keys())
# for i in data:
# 	if(len(i)!=0):
# 		ques.append(i)
print(len(ques))
# ques=[q for q in ques1 if(len(q)!=0)]
# print(len(ques))
# for i in range(len(ques)):
# 	if(len(ques[i])==0):
# 		del ques[i]
# print(len(ques))
tokenizer = RegexpTokenizer(r'\w+')
stop = stopwords.words('english')
dictionary = enchant.request_dict("en_US")
#stemmer = SnowballStemmer("english")

def clean_ques(query):
	# here query is list of words that are present in the question
	query = query.lower()# converted to lowercase alphabet
	query = tokenizer.tokenize(query) # tokenized
	# for i in range(len(ques)):
	# 	if not enchant.dict_exists(ques[i]):
 	# 		ques[i] = dictionary.suggest(sent)[0]
	query = [q for q in query if q not in stop] # removed stop words
	return query
#word1=clean_ques("What are the best places to learn how to dance in Philadelphia?")
#word2=clean_ques("Political Science: What factors tend shift a population in a more liberal or more conservative direction?")
def wordvec(word):
	return model[word]

#Get the Word Centroid Distance
def wcd(sent1, sent2):
	# here sent1 & sent2 both are list of words
	try:
		if(len(sent1)>0 and len(sent2)>0):
			s1 = wordvec(sent1[0])
			s2 = wordvec(sent2[0])
		else:
			return 10000
		for i in range(1,len(sent1)):
			s1 = s1 + wordvec(sent1[i])
		for i in range(1,len(sent2)):
			s2 = s2 + wordvec(sent2[i])
	
		s1 = s1 / len(sent1)
		s2 = s2 / len(sent2)	
		return numpy.linalg.norm(s1 - s2) # returns the norm of the difference of the two vectors	
	except KeyError:
		return 10000
#print(word1)

#Get the Relaxed Word Mover Distance
def rwmd(sent1, sent2):
	s1, s2 = 0, 0
	dist1 , dist2 = 0, 0
	# dist1 is distance to move from sent1 to sent2
	if len(sent1) == 0 or len(sent2) == 0:
		return 0
	for i in range(len(sent1)):
		d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[0]))
		val = 0
		for j in range(len(sent2) - 1):
			if (numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1])) < d): # calculating the minimum distance of sent1[i] with every sent2[j]
				d = numpy.linalg.norm(wordvec(sent1[i]) - wordvec(sent2[j + 1]))
				val = j + 1
		dist1 = dist1 + (1.0 / len(sent1)) * d	

	# dist2 is distance to move from sent2 to sent1	
	for i in range(len(sent2)):
		d = numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[0]))
		val = 0
		for j in range(len(sent1) - 1):
			if (numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[j + 1])) < d):
				d = numpy.linalg.norm(wordvec(sent2[i]) - wordvec(sent1[j + 1]))
				val = j + 1
		dist2 = dist2 + (1.0 / len(sent2)) * d	

	return max(dist1, dist2)			
#print(rwmd(word2,word2))
#print(word2)
#print(rwMd(word1, word1))
#print(rwmd(word1, word2))

def getwcd(query, num):
	print("running function getwcd\nquery= ")
	print(query,num)
	# closest similar num ques chahiye
	dic={}
	for i in range(len(ques)):
		if(len(ques[i])==0):
			continue
		ques1=clean_ques(ques[i])
		val = wcd(query,ques1)
		if(len(dic)<num):
			dic[ques[i]]=val
		else:
			m=max(dic,key=dic.get)
			if(dic[m]>val):
				del dic[m]
				dic[ques[i]]=val
		#create a priority queue to stope the dist
	print("loop finished")
	return list(dic.keys())

def getrwmd(query, kwcd, num):
	print("running function getrwmd\n query= ")
	print(query,num)
	dic={}
	for i in range(len(kwcd)):
		ques1=clean_ques(kwcd[i])
		val=rwmd(query,ques1)
		print (kwcd[i], val)
		if (len(dic)<num):
			dic[kwcd[i]]=val
		else:
			m=max(dic,key=dic.get)
			if(dic[m]>val):
				del dic[m]
				dic[kwcd[i]]=val
	print("loop ended")
	return list(dic.keys())
		#create priority queue to store the dist
	#return top num values	

def getkNN(query, num):
	print("calling getwcd")
	kwcd = getwcd(query, 5 * num)
	print("getwcd call finished")
	print("length of kwcd= "+str(len(kwcd)))
	print("calling getrwmd")
	knn = getrwmd(query, kwcd, num)
	print("getrwmd call finished")
	return knn

def getTagsSimilarQues(query):
	print("calling getTagsSimilarQues")
	knn = getkNN(query, 50)
	print("getkNN call finished")
	for i in range(len(knn)):
		print (	knn[i])
	#print(knn)
	#return tags of all 50 questions returned with count of occurrence	
def main():
	query=input("enter the question")
	query=clean_ques(query)
	print("query cleaned")
	print(query)
	getTagsSimilarQues(query)

if __name__ == '__main__':
	main()