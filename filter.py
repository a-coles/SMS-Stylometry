"""
This program is executed after the RNN is trained. It employs a filter, to decide if the class predicted by the RNN should
be used, or if the google API information should be used. Google API information is used in cases in which the predictions
of the RNN are relatively uncertain.
"""
from keras.models import load_model
import numpy as np
import keras.utils as np_utils
from keras.preprocessing import sequence
from resources.importData import importAndProcess
from non_gt_rnn_3 import createKerasTokens
from sklearn.metrics import accuracy_score

#load in the data and preprocess
messages_train, loc_train, messages_dev, loc_dev, messages_test, loc_test = importAndProcess()
X_train, X_dev, X_test,wordCount = createKerasTokens(messages_train, messages_dev, messages_test)

y_dev = loc_dev
y_dev[loc_dev == 'sg'] = 0
y_dev[loc_dev == 'us'] = 1
y_dev[loc_dev == 'uk'] = 2

# fix random seed for reproducibility
np.random.seed(43)
num_classes = []
# truncate and pad input sequences

max_length = 200
X_dev = sequence.pad_sequences(X_dev, maxlen=max_length)
y_dev = np_utils.to_categorical(y_dev, num_classes)


model = load_model('models/non_gt_rnn_32_model.h5')

predictions = model.predict(X_dev)
pred_class = model.predict_classes(X_dev)

y_int = []
for p in y_dev:
    y = np.argmax(p)
    y_int.append(y)
#Accuracy of the RNN without filtering
print("Accuracy: {}".format(accuracy_score(y_int,pred_class)))


#create dictionary of all textmessages to get API values:
with open('resources/api_messages.txt', 'r') as fp:
    message_probs = fp.readlines()
    message_prob_dict = {}
    for i, line in enumerate(message_probs):
        line = line.strip().split("||")
        message_prob_dict[line[0]] = line[1:]


#goes through the examples of the RNN that were relatively ambiguous
filterSettings = [0.1,0.2,0.3,0.4,0.42,0.45]

for fil in filterSettings:
    pred_class_changed = []
    for p in pred_class:
        pred_class_changed.append(p)
    counter_case = 0
    counter_api = 0
    #get all the uncertain cases:
    for idx,k,p in zip(range(0,len(messages_dev)),pred_class_changed,predictions):
        #in uncertain cases:
        if ((p[0] > fil and p[1] > fil) or
            (p[0] > fil and p[2] > fil) or
            (p[1] > fil and p[2] > fil)):
            #inc counter
            counter_case += 1

            #get google API information for uncertain cases
            #check API
            msg = messages_dev[idx]
            try:
                apiProb = message_prob_dict[msg]
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


    print("Filter: {}, API difference {}%".format(fil,thresh*100))
    print("Number of close cases: {}".format(counter_case))
    print("Number of cases with change: {}".format(counter_api))
    print("Accuracy: {}".format(accuracy_score(y_int, pred_class_changed)))
    print("")



#The part above this is used to determine the best parameters on the development set
#Predicting test set with the best parameters on test set: filter = 0.3; thresh = 0.1

y_test = loc_test
y_test[loc_test == 'sg'] = 0
y_test[loc_test == 'us'] = 1
y_test[loc_test == 'uk'] = 2

# fix random seed for reproducibility
np.random.seed(43)
num_classes = []
# truncate and pad input sequences

max_length = 200
X_test = sequence.pad_sequences(X_test, maxlen=max_length)
y_test = np_utils.to_categorical(y_test, num_classes)


model = load_model('model_32.h5')

predictions = model.predict(X_test)
pred_class = model.predict_classes(X_test)

y_int = []
for p in y_test:
    y = np.argmax(p)
    y_int.append(y)
#Accuracy of the RNN without filtering
print("Accuracy: {}".format(accuracy_score(y_int,pred_class)))


#create dictionary of all textmessages to get API values:
with open('resources/api_messages.txt', 'r') as fp:
    message_probs = fp.readlines()
    message_prob_dict = {}
    for i, line in enumerate(message_probs):
        line = line.strip().split("||")
        message_prob_dict[line[0]] = line[1:]


#goes through the examples of the RNN that were relatively ambiguous
filterSettings = [0.3]

for fil in filterSettings:
    pred_class_changed = []
    for p in pred_class:
        pred_class_changed.append(p)
    counter_case = 0
    counter_api = 0
    #get all the uncertain cases:
    for idx,k,p in zip(range(0,len(messages_test)),pred_class_changed,predictions):
        #in uncertain cases:
        if ((p[0] > fil and p[1] > fil) or
            (p[0] > fil and p[2] > fil) or
            (p[1] > fil and p[2] > fil)):
            #inc counter
            counter_case += 1

            #get google API information for uncertain cases
            #check API
            msg = messages_test[idx]
            try:
                apiProb = message_prob_dict[msg]
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


    print("Filter: {}, API difference {}%".format(fil,thresh*100))
    print("Number of close cases: {}".format(counter_case))
    print("Number of cases with change: {}".format(counter_api))
    print("Accuracy: {}".format(accuracy_score(y_int, pred_class_changed)))
    print("")






