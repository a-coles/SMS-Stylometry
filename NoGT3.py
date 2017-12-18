
import numpy as np
from resources.importData import importAndProcess
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


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


def createKerasTokens(messages_train,messages_dev,messages_test):
    newMsg = preprocess(messages_train)
    toker = Tokenizer(filters='')
    toker.fit_on_texts(newMsg)
    X_train = toker.texts_to_sequences(newMsg)
    print(len(toker.word_counts))
    X_train = np.asarray(X_train)

    newMsg = preprocess(messages_dev)
    X_dev = toker.texts_to_sequences(newMsg)
    X_dev = np.asarray(X_dev)

    newMsg = preprocess(messages_test)
    X_test = toker.texts_to_sequences(newMsg)
    X_test = np.asarray(X_test)

    wordCount = len(toker.word_counts) + 1
    return X_train,X_dev,X_test,wordCount


if __name__ == '__main__':
    messages_train, loc_train, messages_dev, loc_dev, messages_test, loc_test = importAndProcess()

    X_train,X_dev,X_test,wordCount = createKerasTokens(messages_train,messages_dev,messages_test)

    y_train = loc_train
    y_train[loc_train == 'sg'] = 0
    y_train[loc_train == 'us'] = 1
    y_train[loc_train == 'uk'] = 2

    y_dev = loc_dev
    y_dev[loc_dev == 'sg'] = 0
    y_dev[loc_dev == 'us'] = 1
    y_dev[loc_dev == 'uk'] = 2

    y_test = loc_test
    y_test[loc_test == 'sg'] = 0
    y_test[loc_test == 'us'] = 1
    y_test[loc_test == 'uk'] = 2

    # fix random seed for reproducibility
    np.random.seed(43)

    num_classes = []
    # truncate and pad input sequences

    max_length = 200
    X_train = sequence.pad_sequences(X_train, maxlen=max_length)
    X_dev = sequence.pad_sequences(X_dev, maxlen=max_length)
    X_test = sequence.pad_sequences(X_test, maxlen=max_length)

    y_train = np_utils.to_categorical(y_train, num_classes)
    y_dev = np_utils.to_categorical(y_dev, num_classes)
    y_test = np_utils.to_categorical(y_test, num_classes)

    # create the model
    embedding_vector_length = 3
    model = Sequential()
    model.add(Embedding(wordCount, embedding_vector_length, input_length=max_length))
    model.add(LSTM(100,dropout=0.65))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print (model.summary())
    model.fit(X_train, y_train, epochs=10, batch_size=64,verbose=1, validation_data=(X_dev,y_dev), shuffle =True)
    # Final evaluation of the model
    scores = model.evaluate(X_dev, y_dev, verbose=1)
    print ("Accuracy: %.2f%%" % (scores[1]*100))
    model.save('model_3.h5')
