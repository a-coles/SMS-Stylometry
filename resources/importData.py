import json
import numpy as np
from sklearn.model_selection import train_test_split

def importAndProcess():
    np.random.seed(43)
    #Split is train=0.8,dev=0.1,test=0.1
    trainTest = 0.2
    testDev = 0.5


    # Open up pre-processed location corpus
    with open('resources/english_sms_corpus.json', 'r') as fp:
        obj = json.load(fp)

    # Convert the loaded dictionary into pertinent lists
    messages = []
    locations = []
    messages_locations = []
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

    #TODO: Somehow I get different results every time, can you reproduce?
    # With your old version it didn't happen but I don't know whats different

    # Shuffle and split into train and test set
    messages_train, messages_test, loc_train, loc_test = train_test_split(messages, locations, test_size=trainTest,random_state=43)

    # Split the "test" set into development and test set
    messages_dev, messages_test, loc_dev, loc_test = train_test_split(messages_test, loc_test, test_size=testDev,random_state=43)
    return messages_train,loc_train,messages_dev,loc_dev,messages_test,loc_test






