# -*- coding: utf-8 -*-

import os
import json
import re
import numpy as np

# A bit hacky, but works for scraping purposes
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

def test(vectorizer, training_data, training_labels, testing_data, testing_labels):
    train_vectors = vectorizer.fit_transform(training_data)
    test_vectors = vectorizer.transform(testing_data)

    # Set up the Naive Bayes classifier
    mnb = MultinomialNB(alpha=1.3)
    mnb.fit(train_vectors, training_labels)
    prediction = mnb.predict(test_vectors)

    # How good was it?
    accuracy = mnb.score(test_vectors, testing_labels)
    print "Accuracy: {}".format(accuracy)

if __name__ == '__main__':
    # Open up pre-processed location corpus
    with open('english_sms_corpus.json', 'r') as fp:
        obj = json.load(fp)

    # Convert the loaded dictionary into pertinent lists
    messages = []
    locations = []
    messages_locations = []
    stop_words = set(stopwords.words("english"))
    for message, location in obj.items():
        messages.append(message)

        # Clean up the location
        if location.lower() == "singapore":
            location = "sg"
        elif location.lower() == "united states":
            location = "us"
        elif location.lower() == "united kingdom":
            location = "uk"
        locations.append(location)

    # Convert these to a numpy array
    messages = np.array(messages)
    locations = np.array(locations)

    # Shuffle and split into train and test set
    messages_train, messages_test, loc_train, loc_test = train_test_split(messages, locations, test_size=0.2, random_state=42)

    # Split the "test" set into development and test set
    messages_dev, messages_test, loc_dev, loc_test = train_test_split(messages_test, loc_test, test_size=0.5, random_state=42)

    # Vectorize the message
    vectorizer = CountVectorizer(ngram_range=(1,1), decode_error='replace', min_df=0.0001, stop_words=stop_words)

    # Uncomment to test the development set
    #test(vectorizer, messages_train, loc_train, messages_dev, loc_dev)

    # Test the testing set
    test(vectorizer, messages_train, loc_train, messages_test, loc_test)
