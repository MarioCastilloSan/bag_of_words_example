# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 15:49:06 2023

@author: Mario
"""


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_table('Restaurant_Reviews.tsv')



#Text Cleaning
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
potSt = PorterStemmer()


def commentsListData():
    commentsList=[]
    for i in range(len(data['Review'])):
        review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
        review = review.lower()
        review = review.split()
        review = [potSt.stem(word) for word in review if not word in set(stopwords.words('english'))]
        review = ' '.join(review)
        commentsList.append(review)
    return commentsList

commentsList = commentsListData()
    
    
#Creating the bag of words
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(commentsList).toarray()
y = data.iloc[:, 1].values


# divide data for training and validation
from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 0.20, random_state = 0)
 



# SVM with rbf
from sklearn.svm import SVC
classifier = SVC(C =10,gamma= 0.1,kernel = "rbf", random_state = 0)
classifier.fit(xTrain, yTrain)

yPred  = classifier.predict(xTest)





def getPrediction(text:str):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [potSt.stem(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)
    text = cv.transform([text]).toarray()
    predictedCategory = classifier.predict(text)
    predictedLabel = "Positive" if predictedCategory == 1 else "Negative"
    return predictedLabel






# confussion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(yTest, yPred)

#Values Section
TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]

Accuracy = (TP+TN)/(TP+TN+FP+FN)
Recall = TP/(TP+FN)
F1Score = 2*Accuracy*Recall/(Accuracy+Recall)




#K-Fold Cross Validation
from sklearn.model_selection import cross_val_score
cvs = cross_val_score(estimator = classifier, X = xTrain, y= yTrain, cv = 10)
cvs.mean()
cvs.std()

def getMetrics():
    return Accuracy, Recall, F1Score, cvs.std()

def getData():
    return data






