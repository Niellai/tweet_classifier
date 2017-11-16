
# coding: utf-8

# In[42]:

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

import import_notebook
from readWrite import ReadWrite


# In[66]:

class TweetClassifier(object):
    readWrite = ReadWrite()
    xdata = []
    ydata = []
    gs_clf = []
    
    def __init__(self):
        tweetFile = "combine_tweets.txt"
        tweetClassFile = "combineVectorsResult.txt"
        self.xdata = self.readWrite.readOriFile(tweetFile)
        self.ydata = self.readWrite.readFileClassifier(tweetClassFile)
        self.xdata = np.array(self.xdata)
        self.ydata = np.array(self.ydata)
        print("xdata: {}".format(self.xdata.shape))
        print("ydata: {}".format(self.ydata.shape))
    
    def gridSearchFit(self):
        text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
        text_clf = text_clf.fit(self.xdata, self.ydata)
        
        parameters = {'vect__ngram_range': [(1, 1), (1, 2)], 'tfidf__use_idf': (True, False), 'clf__alpha': (1e-2, 1e-3)}
        self.gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
        self.gs_clf = self.gs_clf.fit(self.xdata, self.ydata)
        
    def predict(self, sentence):
        arr = []
        arr.append(sentence)
        predicted = self.gs_clf.predict(arr)
        self.printPredict(predicted, arr)
        return predicted
    
    def predictDataset(self, dataset):
        predicted = self.gs_clf.predict(dataset)
        self.printPredict(predicted, dataset)
        return predicted
        
    def printPredict(self, predicted, data):
        for idx, score in enumerate(predicted):    
            if(score > 0): 
                print(score)
                print(data[idx])


# In[72]:

# tweetClassifier = TweetClassifier()
# tweetClassifier.gridSearchFit()


# In[73]:

# readWrite = ReadWrite()
# tweetLive = "user_tweet_live.txt"    
# livedata = readWrite.readOriFile(tweetLive)
# livedata = np.array(livedata)
        
# predicted = tweetClassifier.predictDataset(livedata)
# predicted = tweetClassifier.predict("@SMRT_Singapore signal fault, add 20min travel time from Joo koon to chinese garden.")

