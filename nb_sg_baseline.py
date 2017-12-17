# -*- coding: utf-8 -*-

import numpy as np
from resources.importData import importAndProcess
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score

# A bit hacky, but works for scraping purposes
# encoding=utf8

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def testNB(vectorizer, training_data, training_labels, testing_data, testing_labels):
    train_vectors = vectorizer.fit_transform(training_data)
    test_vectors = vectorizer.transform(testing_data)

    # Set up the Naive Bayes classifier
    mnb = MultinomialNB(alpha=1.3)
    mnb.fit(train_vectors, training_labels)

    predictionTrain = mnb.predict(train_vectors)
    predictionTest = mnb.predict(test_vectors)

    # How good was it?
    accuracy = accuracy_score(training_labels, predictionTrain)
    print("\n-\tNaive Bayes Classifier\t-")
    print("Train set:")
    print("Accuracy: {}".format(accuracy))

    accuracy = accuracy_score(testing_labels, predictionTest)
    print ("\n-\tNaive Bayes Classifier\t-")
    print ("Test set:")
    print ("Accuracy: {}".format(accuracy))

def guessSG(testing_labels):
    prediction = []
    for i in testing_labels:
        #always guess singapore
        prediction.append("sg")
        # How good was it?
    accuracy = accuracy_score(testing_labels, prediction)
    print("-\tAlways Guess Singapore\t-")
    print("Accuracy: {}".format(accuracy))

if __name__ == '__main__':
    np.random.seed(43)

    stop_words = set(stopwords.words("english"))
    messages_train, loc_train, messages_dev, loc_dev, messages_test, loc_test = importAndProcess()

    # Vectorize the message
    vectorizer = CountVectorizer(ngram_range=(1,1), decode_error='replace', min_df=0.0001, stop_words=stop_words)

    # Uncomment to test the development set
    #testNB(vectorizer, messages_train, loc_train, messages_dev, loc_dev)

    # Test the testing set
    testNB(vectorizer, messages_train, loc_train, messages_test, loc_test)
    #Always guessing Singapore baseline
    print("\nTrain set accuracy:")
    guessSG(loc_train)
    print("Test set accuracy:")
    guessSG(loc_test)

