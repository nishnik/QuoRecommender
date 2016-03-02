import os,json
import requests
from bs4 import BeautifulSoup
dic={}
count=0
for i in os.listdir('/home/adarsh/Documents/DA/Quora_logs'):
	filename="/home/adarsh/Documents/DA/Quora_logs/"+i
	#print filename
	soup=BeautifulSoup(open(filename))
	question = soup.find_all("div",{"class":"revision"})
	try:
		q=question[-1]
		#print q.getText()
		q=q.getText()
		q=q.encode('utf-8')
		print q
		topic=soup.find_all("span",{"class":"TopicName"})
		topics=[]
		for i in topic:
			topics.append((i.getText()).encode('utf-8'))
		#print topics
		try:
			dic[q]=topics
		except UnicodeEncodeError:
			print "can't write"
	except IndexError:
		pass
filehandle=open("text.txt","w")
json.dump(dic,filehandle,indent=2)
filehandle.close()