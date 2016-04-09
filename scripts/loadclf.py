from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

vectorizer = joblib.load('pkl/Tfidf.pkl')
question = ["Formula 1: How does the Formula 1 scoring system work?"]
print(question)
question = vectorizer.transform(question);

mlb = joblib.load('pkl/Multi.pkl')

knnclf = joblib.load('pkl/KNeig.pkl')
pred = knnclf.predict(question)
pred = mlb.inverse_transform(pred)
print("KNN prediction: ", pred)

ovrclf = joblib.load('pkl/OneVs.pkl')
pred = ovrclf.predict(question)
pred = mlb.inverse_transform(pred)
print("OVR prediction: ", pred)
