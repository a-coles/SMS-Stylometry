# -*- coding: utf-8 -*-

import os
import json
import re

# A bit hacky, but works for scraping purposes
# encoding=utf8
import sys
reload(sys)
sys.setdefaultencoding('utf8')

from nltk.corpus import stopwords

# ----------------------------------------------------------------------------
# This file calculates the "Google baseline" for the English SMS location corpus.
# For each message, it removes stop words, and for the remaining words, it
# collects the probability of that word under each possible location, sums
# these probabilities according to location for each word in the message and
# divides by the length of the message (without stop words). It thus gets an
# average probability of the message belonging to each location.
#
# The baseline is calculated by choosing the highest average probability, and
# choosing the location that generated it.
# ----------------------------------------------------------------------------



if __name__ == '__main__':
    stop_words = set(stopwords.words("english"))

    # Open up pre-processed location corpus and format
    with open('english_sms_corpus.json', 'r') as fp:
        message_locations = json.load(fp)
    for message, location in message_locations.items():
        if location.lower() == "singapore":
            message_locations[message] = "sg"
        elif location.lower() == "united states":
            message_locations[message] = "us"
        elif location.lower() == "united kingdom":
            message_locations[message] = "uk"

    message_locations = {k.decode('utf8'): v.decode('utf8') for k, v in message_locations.items()}


    if not os.path.exists('api_messages.txt'):
        # Get the average probability of the sentence per location
        # and save it down
        with open ('api_words.txt', 'r') as inf, open('api_messages.txt', 'w') as outf:
            api_words = inf.readlines()
            for message, location in message_locations.items():
                full_message = message
                print full_message

                # Remove stop words
                message = []
                for word in full_message.split():
                    if word not in stop_words:
                        message.append(word)

                # Get probs of each word in lists by location
                uk_list, us_list, sg_list = [], [], []
                for word in message:
                    for i, line in enumerate(api_words):
                        if i == 0 or i == 1:
                            continue
                        if word in line:
                            line = line.split("||")
                            uk_list.append(float(line[1]))
                            us_list.append(float(line[2]))
                            sg_list.append(float(line[3]))
                            break

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

                new_line = "{}||{}||{}||{}\n".format(full_message, uk_prob, us_prob, sg_prob)
                outf.write(new_line)


    # Open up messages-probabilities file and format
    with open('api_messages.txt', 'r') as fp:
        message_probs = fp.readlines()
    message_prob_list = []
    for i, line in enumerate(message_probs):
        line = line.strip().split("||")
        message_prob_list.append(line)


    # Get "average API baseline"
    total_counter = 0
    correct_counter = 0
    for line in message_prob_list:
        message = unicode(line[0], "utf-8").strip()

        # If probabilities weren't assigned, skip the message
        if message == "":
            continue
        if len(line) < 4 or len(line) > 4:
            continue

        total_counter = total_counter + 1

        probs_by_loc = {line[1]:"uk", line[2]:"us", line[3]:"sg"}
        probs = [line[1], line[2], line[3]]
        max_prob = max(probs)
        guess = probs_by_loc[max_prob]
        try:
            truth = message_locations[message]
        except KeyError:
            print "this message wasn't found, probably too short: " + str(message)

        if guess == truth:
            print "Correct guess!"
            print "Message: {}".format(message)
            print "Location: {} {}".format(guess, max_prob)
            correct_counter = correct_counter + 1

    # Print out accuracy
    accuracy = float(correct_counter)/float(total_counter)
    print "Total accuracy for API baseline: {}".format(accuracy)
