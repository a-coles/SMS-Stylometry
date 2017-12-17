# -*- coding: utf-8 -*-

import json
import re
import pandas
import numpy as np
import time
import datetime

# A bit hacky, but works for scraping purposes
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from nltk.corpus import stopwords
from pytrends.request import TrendReq
from pytrends import exceptions
from pytrends.request import exceptions

# ----------------------------------------------------------------------------
# This file gets the probabilities of each word used in every message per
# location, i.e. given a message "I like cats", it gets the probability of "I"
# belonging to a speaker of English from the United Kingdom , the United
# States, and Singapore; and so on for "like" and "cats".
#
# The Google Trends database is queried on each word, and the resulting
# probabilities are saved so that words are not repeatedly queried.
# ----------------------------------------------------------------------------

if __name__ == '__main__':
    # Open up pre-processed location corpus
    with open('english_sms_corpus.json', 'r') as fp:
        obj = json.load(fp)

    # Convert the loaded dictionary into pertinent lists
    messages = []
    locations = []
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

    # Get text files ready
    if not os.path.exists('api_words.txt'):
        with open('api_words.txt', 'w') as outf:
            outf.write("API WORDS\n")
            outf.write("word||uk||us||sg\n")
    if not os.path.exists('api_messages.txt''):
        with open('api_messages.txt', 'w') as outf:

    # Get "average API baseline"
    total_counter = 0
    correct_counter = 0
    location_prob_dict = {}
    word_prob_dict = {}
    with open('api_messages.txt', 'r') as api_messages:
        api_messages = api_messages.readlines()
    with open('api_words.txt', 'r') as api_words:
        api_words = api_words.readlines()
    beg = time.time()
    hour = beg + 3600
    for i, message in enumerate(messages):

        checker1 = False
        checker2 = False
        for line in api_messages:
            if message in line:
                checker1 = True
            if checker1:
                checker2 = True
                break
        if checker2:
            continue


        total_counter = total_counter + 1
        print "Now processing message {} of {}...".format(i, len(messages))

        # Clean up the message
        full_message = message
        print full_message
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

        # If message is empty, probs are zero
        if message == []:
            uk, us, sg = 0.0, 0.0, 0.0
            continue

        # Get probabilities of each sentence
        uk_list, us_list, sg_list = [], [], []
        for word in message:
            print "   Processing word {}...".format(word)

            # If we already have probability for this word stored,
            # use that and don't query it
            checker3 = False
            checker4 = False
            for line in api_words:
                if word in line:
                    checker3 = True
                    line = line.split("||")
                    uk = line[1]
                    us = line[2]
                    sg = line[3]
                if checker3:
                    checker4 = True
                    break
            if checker4:
                continue

            # Otherwise, query the Google Trends API.
            print "     didn't find, so querying."
            kw_list = [word]
            try:
                pytrends = TrendReq(hl='en-US', tz=360)
                pytrends.build_payload(kw_list, cat=0, timeframe="today 5-y", geo='', gprop='')
            except exceptions.ResponseError as e:
                # Google doesn't like us querying this much, so space out next requests
                print "     " + str(e)
                print "Going to sleep for a while, then will re-query."
                time.sleep(300)
                ready = False
                while not ready:
                    try:
                        pytrends = TrendReq(hl='en-US', tz=360)
                        pytrends.build_payload(kw_list, cat=0, timeframe="today 5-y", geo='', gprop='')
                    except exceptions.ResponseError as e:
                        print "     " + str(e)
                        print "Still not ready."
                        time.sleep(300)
                    else:
                        ready = True
            except exceptions.SSLError as e:
                print "     " + str(e)
                uk, us, sg = 0.0, 0.0, 0.0

            # Try to query it by country
            term_not_found = False
            try:
                loc_table = pytrends.interest_by_region(resolution='COUNTRY')
            except KeyError:
                # Nobody says it, apparently, as it's not in the index for these
                # countries, so probs go to zero
                uk, us, sg = 0.0, 0.0, 0.0
                term_not_found = True
            except exceptions.SSLError as e:
                print "     " + str(e)
                uk, us, sg = 0.0, 0.0, 0.0

            # Get counts for query by each country (100 is top, so
            # they act like percentages already)
            if not term_not_found:
                try:
                    uk = float(loc_table.loc["United Kingdom", word])/float(100)
                except KeyError:
                    uk = 0.0
                try:
                    us = float(loc_table.loc["United States", word])/float(100)
                except KeyError:
                    us = 0.0
                try:
                    sg = float(loc_table.loc["Singapore", word])/float(100)
                except KeyError:
                    sg = 0.0

            # Save the word we just queried so we don't have to query it again
            line = "{}||{}||{}||{}\n".format(word, uk, us, sg)
            api_words.append(line)
            with open('api_words.txt', 'a') as outf:
                outf.write(line)

            uk_list.append(uk)
            us_list.append(us)
            sg_list.append(sg)

        try:
            uk_prob = float(sum(uk_list))/len(message)
        except ZeroDivisionError:
            uk_prob = 0.0
        try:
            us_prob = float(sum(us_list))/len(message)
        except ZeroDivisionError:
            us_prob = 0.0
        try:
            sg_prob = float(sum(sg_list))/len(message)
        except ZeroDivisionError:
            sg_prob = 0.0

        # Save the
        with open('api_messages.txt', 'a') as outf:
            line = "{}||{}||{}||{}\n".format(full_message, uk_prob, us_prob, sg_prob)
            outf.write(line)
