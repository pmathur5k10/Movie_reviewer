
import os
#used to format operating system command line instructions 
from sklearn.feature_extraction.text import CountVectorizer
# importing Vectorizer from sckit learn library
from sklearn.ensemble import RandomForestClassifier
# importing Random Forest Algorithm from sckit library
import re
# library for text string functions
from bs4 import BeautifulSoup
#beautiful soup removes html tags

import nltk
#library for natural language processing

from nltk.corpus import stopwords 
import pandas as pd 
# used to parse csv files
import numpy as np 
#scientific library for python containing multidimension array and probablities and fourier transforms




#function to clean review data

def review_to_wordlist( review, remove_stopwords=False ):
        # Function to convert a document to a sequence of words,
        # optionally removing stop words.  Returns a list of words.
        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case and split them
        words = review_text.lower().split()
        #
        # 4. Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            words = [w for w in words if not w in stops]
        #
        # 5. Return a list of words
        return(words)




#                                                                          FETCHING DATA 

#if __file__=='__main__':
train=pd.read_csv(os.path.join(os.path.dirname('__main__'),'data','C:\ml\Data\labeledTrainData.tsv'),header=0,delimiter="\t",quoting=3)
test=pd.read_csv(os.path.join(os.path.dirname('__main__'),'data','C:\ml\Data\TestData.tsv'),header=0 ,delimiter="\t",quoting =3)

#train.head(3)


#train=pd.read_csv('C:\\ml\\unlabeledTrainData.csv',header=None)


#if __file=__main__  helps the python to differntiate between a script and a module 
# When this script is run, then the the following code is read but if someone else uses this file as amodule in another program then this csv file may not be read

print ('sample review......')
print (train["review"][0])	

#raw_input('Enter to continue')

#nltk.download()

#                                                                           CLEANING DATA


clean_train_review=[]

for i in xrange(0,len(train["review"])):
	clean_train_review.append(" ".join(review_to_wordlist(train["review"][i],True)))


# create a new list to store the clean review
# loop over each review using xrange( used for gigantic list that is memory sensitive) and then use the KaggleWeod2Vec library function
#   review_to_wordlist to clean the review and append them to the new list


#                                                                            CREATE A BAG OF WORDS I.E. VECTORIZER


Vectorizer=CountVectorizer( analyzer="word" ,tokenizer= None ,preprocessor=None,stop_words=None ,max_features=5000)

#initialzed a vectorizer for 500 words
train_data_features=Vectorizer.fit_transform(clean_train_review)
#fitting the clean review data into the vectorizer
train_data_features=train_data_features.toarray()
#converting the data to an array 
#PROTIP := Numpy arraysa are easier to use 


#                                                                              TRAINING DATA


forest=RandomForestClassifier(n_estimators=100)
#initialise a random forest with 100 trees
forest=forest.fit(train_data_features,train["sentiment"])
#fit the vectorised data into the forest along with the sentiment of clean train data

#                                                                         PREPARING TEST DATA

clean_test_data=[]

for i in xrange(0,len(test["review"])):
	clean_test_data.append(" ".join(review_to_wordlist(test["review"][i],True)))

test_data_feature=Vectorizer.transform(clean_test_data)
test_data_feature=test_data_feature.toarray()

#created a  new test data list
#looped through all the test data reviews and use KaggleWord2Vec library to clean the data
# vectorized the clean data and converted into an array



#                                                                         PREDICT DATA

result=forest.predict(test_data_feature)
# calling the random forest we created to predict the test data result

#                                                                   OUTPUT INTO CSV FILE


output=pd.DataFrame(data={"id":test["id"],"sentiment":result})
#calling pandas to convert the data into a csv format
output.to_csv(os.path.join(os.path.dirname('__main__'),'data','C:\ml\Movie_Sentiment.csv'),index=False,quoting=3)
print("Data written to csv.")
#                                                                         END






