import json
import re
import pandas
import numpy as np

from nltk.corpus import stopwords
from pytrends.request import TrendReq

if __name__ == '__main__':
    # Open up pre-processed location corpus
    with open('english_sms_corpus.json', 'r') as fp:
        obj = json.load(fp)

    # Convert the loaded dictionary into pertinent lists
    messages = []
    locations = []
    stop_words = set(stopwords.words("english"))
    for message, location in obj.items():
        # Clean up the message
        message = message.strip().lower()
        message = re.sub(r"[^A_Za-z0-9\s]", "", message)
        message = re.sub(r"\s[0-9]\s", " ", message)
        message = message.split()
        # Remove stop words
        message2 = []
        for word in message:
            if word not in stop_words:
                message2.append(word)
        message = message2
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

    # Access Google Trends API
    pytrends = TrendReq(hl='en-US', tz=360)

    # Get "average API baseline"
    total_counter = 0
    correct_counter = 0
    for i, message in enumerate(messages):
        total_counter = total_counter + 1
        # Query the API
        uk_list, us_list, sg_list = [], [], []
        for word in message:
            kw_list = [word]
            pytrends.build_payload(kw_list, cat=0, timeframe="today 5-y",
                                   geo='', gprop='')
            try:
                loc_table = pytrends.interest_by_region(resolution='COUNTRY')
            except:
                # Nobody says it, apparently
                uk, us, sg = 0.0, 0.0, 0.0
            # Get counts for query by each country (100 is top, so
            # they act like percentages already)
            try:
                uk = float(loc_table.loc["United Kingdom", word])/float(100)
            except:
                uk = 0.0
            try:
                us = float(loc_table.loc["United States", word])/float(100)
            except:
                us = 0.0
            try:
                sg = float(loc_table.loc["Singapore", word])/float(100)
            except:
                sg = 0.0
            uk_list.append(uk)
            us_list.append(us)
            sg_list.append(sg)
        uk_prob = float(sum(uk_list))/len(message)
        us_prob = float(sum(us_list))/len(message)
        sg_prob = float(sum(sg_list))/len(message)
        labeled_probs = {uk_prob:"uk", us_prob:"us", sg_prob:"sg"}
        max_prob = max([uk_prob, us_prob, sg_prob])
        guess = labeled_probs[max_prob]
        truth = locations[i]

        if guess == truth:
            print "Correct guess!"
            print "Message: {}".format(message)
            print "Location: {} {}".format(guess, max_prob)
            correct_counter = correct_counter + 1

    accuracy = float(correct_counter)/float(total_counter)
    print "Total accuracy for API baseline: {}".format(accuracy)
