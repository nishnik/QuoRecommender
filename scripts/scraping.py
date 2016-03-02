import requests
from bs4 import BeautifulSoup

# Put File path here
FILE = ""

soup=BeautifulSoup(open(FILE))
question = soup.find_all("a",{"class":"question_link","target":"_top"})
for link in question:
	print link.get("href")[1:]
"""
topic=[]
print topic
c=soup.find_all("span",{"class":"TopicNameSpan TopicName"})
for t in c:
	s=t.getText()
	if s not in topic:
		#print str(s),type(s)
		topic.append(str(s))
print topic
"""
children=soup.find_all("div",{"class":"QuestionTopicListItem TopicListItem topic_pill"})
newtopic=[]
for child in children:
	s=child.find_all("span",{"class":"TopicNameSpan TopicName"})
	for q in s:
		newtopic.append(str(q.getText()))
print newtopic
