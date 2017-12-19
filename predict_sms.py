import numpy as np
import re

from pytrends.request import TrendReq
from pytrends import exceptions
from pytrends.request import exceptions

from resources.importData import importAndProcess
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.models import load_model

from nltk.corpus import stopwords

def preprocess(messages):
    # get single words out of sentences
    tokMessages = []

    maxlength = 0
    longest = []
    for msg in messages:
        tok = word_tokenize(msg)
        tokMessages.append(tok)
        if len(tok) > maxlength:
            maxlength = len(tok)
            longest = tok

    newMsg = []
    for msg in tokMessages:
        n = " ".join(msg)
        newMsg.append(n)
    return newMsg

def get_location(dist):
    index = np.argmax(dist)
    if index == 0:
        return "Singapore"
    elif index == 1:
        return "American (United States)"
    elif index == 2:
        return "British (United Kingdom)"

def google_message_prob(message):
    with open('resources/api_words.txt', 'r') as api_words:
        api_words = api_words.readlines()

    # Clean up the message
    full_message = message
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

    # Get probabilities of the message
    uk_list, us_list, sg_list = [], [], []
    for word in message:
        #print "   Processing word {}...".format(word)

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
        #print "     didn't find, so querying."
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
        with open('resources/api_words.txt', 'a') as outf:
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

    # Return the probability list
    # [uk, us, sg]
    return [uk_prob, us_prob, sg_prob]

if __name__ == '__main__':
    # Get SMS from user
    input_sms = [raw_input("Please type your SMS and press enter:\n")]
    print "Thank you. Now feeding it to the RNNs to predict your dialect of English..."
    stop_words = set(stopwords.words("english"))
    #input_sms = ["Here's the plan... we both pool together a bunch of money and buy lotto tickets. When we win, we won't tell nobody and go hideout in sunny Mexico."]

    toker = Tokenizer(filters = '')
    toker.fit_on_texts(input_sms)

    #newMsg = preprocess(input_sms)
    newMsg = input_sms
    oldMsg = newMsg[0]
    newMsg = toker.texts_to_sequences(newMsg)
    newMsg = np.asarray(newMsg)

    # truncate and pad input sequences
    max_length = 200
    newMsg = sequence.pad_sequences(newMsg, maxlen=max_length)

    gt_embedding_model = load_model('models/gt_embedding_model.h5')
    non_gt_3_model = load_model('models/non_gt_rnn_3_model.h5')
    non_gt_32_model = load_model('models/non_gt_rnn_32_model.h5')

    gt_embedding_pred = get_location(gt_embedding_model.predict(newMsg))
    non_gt_3_pred = get_location(non_gt_3_model.predict(newMsg))
    non_gt_32 = non_gt_32_model.predict(newMsg)
    non_gt_32_pred = get_location(non_gt_32)
    pred_class = non_gt_32_model.predict_classes(newMsg)


    # Now, deal with the GT-Filter RNN

    # Get the Google Trends probabilities for each word in the message
    google_message_probs = google_message_prob(oldMsg)

    #goes through the examples of the RNN that were relatively ambiguous
    filterSettings = [0.3]

    for fil in filterSettings:
        pred_class_changed = []
        for p in pred_class:
            pred_class_changed.append(p)
        counter_case = 0
        counter_api = 0
        #get all the uncertain cases:
        for idx,k,p in zip(range(0,1),pred_class_changed,non_gt_32):
            #in uncertain cases:
            if ((p[0] > fil and p[1] > fil) or
                (p[0] > fil and p[2] > fil) or
                (p[1] > fil and p[2] > fil)):
                #inc counter
                counter_case += 1

                #get google API information for uncertain cases
                #check API
                msg = newMsg[idx]
                try:
                    #apiProb = message_prob_dict[msg]
                    apiProb = google_message_probs
                except KeyError:
                    apiProb = [0.0, 0.0, 0.0]

                apiProb = [float(a) for a in apiProb]
                if apiProb != [0.0,0.0,0.0]:
                    #only look at the two highest rated cases by the RNN
                    apiHigh = []
                    for i,a in enumerate(apiProb):
                        if i != np.argmin(p):
                            apiHigh.append((i,a))

                    #only use API if the best value is at least better by thresh than the next
                    #this avoids situations where both the best values are similar
                    thresh = 0.10

                    [(i1,a1),(i2,a2)] = apiHigh
                    if (a1 - (thresh*a1) > a2):
                        pred_class_changed[idx] = i1
                        counter_api += 1
                    elif (a2 - (thresh*a2) > a1):
                        pred_class_changed[idx] = i2
                        counter_api += 1
            filter_pred = get_location(pred_class_changed[idx])

    print "The GT-Embedding RNN predicts: {}".format(gt_embedding_pred)
    print "The Non-GT RNN (dim. 3) predicts: {}".format(non_gt_3_pred)
    print "The Non-GT RNN (dim. 32) predicts: {}".format(non_gt_32_pred)
    print "The GT-Filter RNN predicts: {}".format(filter_pred)
