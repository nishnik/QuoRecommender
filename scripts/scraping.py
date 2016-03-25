# To run in DEBUG mode, add -O flag, i.e python -O scraping.py
import os,json
import requests
from bs4 import BeautifulSoup
dic={}
count=0

PATH = "../logs/"
for i in os.listdir(PATH):
	filename = PATH + i
	soup = BeautifulSoup(open(filename))
	question = soup.find_all("div", {"class":"revision"})
	try:
		q = question[-1]
		if __debug__:
			print q.getText()
		q = q.getText()
		q = q.encode('utf-8')
		print q
		topic = soup.find_all("span",{"class":"TopicName"})
		topics = []
		for i in topic:
			topics.append((i.getText()).encode('utf-8'))
		if __debug__:
			print topics
		try:
			dic[q]=topics
		except UnicodeEncodeError:
			print "Can't write"
	except IndexError:
		pass
filehandle = open("tags.json","w")
json.dump(dic,filehandle,indent = 2)
filehandle.close()
