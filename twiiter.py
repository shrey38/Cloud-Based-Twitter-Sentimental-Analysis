"""Created by Shreeyash Pandey
"""
import pickle

import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#get all the stop wrods present
stop_words=stopwords.words('english')

twit_dataset=pd.read_csv(r"C:\Twitter_Sentimental_analysis")

emojis={
    ':)':'smile',':-)':'smile',':(':'sad',';d':'wink',':-(':'sad',':@':'shocked',':\\':'annoyed',';)':'wink','O.o':'confused',':-@':'socked'
}

#to clean te data and convert all the data to lowercase
def clean_data(data):
    data=str(data).lower()
    data=re.sub(r"@\S",r'',data)

    for emoji in emojis.keys():
        data=data.replace(emoji,emojis[emoji])

    data=re.sub("\s+"," ",data)
    data = re.sub("\n+", " ", data)

    letters=re.sub("[^a-zA-Z]"," ",data)
    return letters;

#To remove all the stop words
def stop_words(words):
    filter=[]
    for w in words:
        if w not in stop_words:
            filter.append(w)
    return filter

def load_model():
    vector_file=open("C:\Twitter_Sentimental_analysis/saved_model")
    vector=pickle.load(vector_file)
    vector_file.close()
    model_file=open("C:\Twitter_Sentimental_analysis/saved_model")
    model=pickle.load(model_file)
    model_file.close()

    return vector,model

def predict(vector,model,text):
    final_data=vector.transform([clean_data(text)])
    sentiment=model.predict(final_data);
    return sentiment

from flask import Flask,request
import json

app=Flask(__name__)

@app.route("/predict",methods=['POST'])

def main():
    req=request.data
    vector,model=load_model()
    sentiment=predict(vector,model,req)

    return sentiment[0]

if __name__="__main__"
    app.run(debug=True)


twit_dataset['text']=twit_dataset['text'].apply(lambda x:clean_data(x))

twit_dataset['text']=twit_dataset['text'].apply(lambda x:x.split(" "))
twit_dataset['text']=twit_dataset['text'].apply(lambda x:stop_words(x))

lemmit=WordNetLemmatizer()
twit_dataset['text']=twit_dataset['text'].apply(lambda x:[lemmit.lemmatize(word) for word in x])
twit_dataset['text']=twit_dataset['text'].apply(lambda x:' '.join(x))


from sklearn.model_selection import train_test_split

train,test=train_test_split(twit_dataset,test_size=0.2,random_state=42)

x_train=train['text']
x_test=test['text']

from sklearn.feature_extraction.text import TfidfVectorizer

vector=TfidfVectorizer(use_idf=True)
x_train=vector.fit_transform(x_train)

x_test=vector.transform(x_test)


from sklearn.metrics import accuracy_score,precision_score,recall_score
def model_per(model):
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_pred,test['sentiment'])
    recall=recall_score(y_pred,test['sentiment'],pos_label="negative")
    precision=precision_score(y_pred,test['sentiment'],pos_label="negative")
    return (accuracy,recall,precision)


from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

BNBmodel=BernoulliNB(alpha=2)
SVCmodel=LinearSVC()
LRmodel=LogisticRegression(C=2,max_iter=1000,n_jobs=-1)

models=[BNBmodel,SVCmodel,LRmodel]
model_scores={}
model_fitted={}
for model in models:
    model.fit(x_train,train['sentiment'])
    accurecy=model_per(model)
    model_scores[model.__class__.__name__]=accurecy[1]
    model_fitted[model.__class__.__name__] =model
best_model=max(model_scores,key=model_scores.get)

filename=best_model+'.pickle'
import pickle

with open(r"C:\Twitter_Sentimental_analysis/saved_model", 'wb') as model1:
     pickle.dump(model_fitted[best_model],model1)
with open("saved_model/tfvectorizer.pickle",'wb') as file:
     pickle.dump(vector,file)

