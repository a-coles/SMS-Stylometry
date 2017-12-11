"""
This is used to see which messages were classified wrongly. Additionally the google api prediction is displayed
I made this file to see if the filter idea we thought of might work or not.
"""
from keras.models import load_model
import numpy as np
import keras.utils as np_utils
from keras.preprocessing import sequence
from resources.importData import importAndProcess
from KerasRNN import createKerasTokens
from keras.preprocessing.text import Tokenizer


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


model = load_model('model.h5')

predictions = model.predict(X_dev)
pred_class = model.predict_classes(X_dev)


#create dictionary of all textmessages to get API values:
with open('resources/api_messages.txt', 'r') as fp:
    message_probs = fp.readlines()
    message_prob_dict = {}
    for i, line in enumerate(message_probs):
        line = line.strip().split("||")
        message_prob_dict[line[0]] = line[1:]





for k in zip(range(0,len(messages_dev)),pred_class,y_dev,predictions):
    (idx, pred, y, probs) = k
    #print(pred,y)
    if pred != np.argmax(y):
        #print("Index {}".format( idx))
        msg = messages_dev[idx]
        try:
            apiProb = message_prob_dict[msg]
        except KeyError:
            apiProb = [0.0,0.0,0.0]

        print("Prediction: {}, Real: {}".format(pred,y))
        print("NN Probs: {}".format(probs))
        print("API Probs: {}".format(apiProb))
        print("Message: {}".format(messages_dev[idx]))
        print("\n")



