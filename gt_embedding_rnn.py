import json
import numpy as np
from resources.importData import importAndProcess

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.preprocessing import sequence
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils

from keras.models import load_model
from keras.models import Sequential

from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding

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


if __name__ == '__main__':
    messages_train, loc_train, messages_dev, loc_dev, messages_test, loc_test = importAndProcess()

    newMsg = preprocess(messages_train)
    toker = Tokenizer(filters = '')
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

    # Get "location" embeddings
    with open('resources/api_words.txt', 'r') as fp:
        vocab_weights = {}
        lines = fp.readlines()[2:]
        for line in lines:
            line = line.strip()
            line = line.split("||")
            weights = [float(w) for w in line[1:]]
            vocab_weights[line[0]] = weights

    embedding_matrix = []
    for i in range(0, len(toker.word_counts)+1):
        embedding_matrix.append([0.0, 0.0, 0.0])

    # Also save down the word-to-index mapping
    with open('resources/word_indicies.txt', 'w') as fp:
        vocabulary = toker.word_index.keys()
        for word in vocabulary:
            if word in vocab_weights:
                index = toker.word_index[word]
                fp.write('{},{}'.format(word, index))
                fp.write('\n')
                embedding_matrix[index] = vocab_weights[word]

    embedding_matrix = np.array(embedding_matrix)

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

    embedding_vector_length = 3
    model = Sequential()
    embedding_layer = Embedding(len(toker.word_counts)+1, embedding_vector_length, weights=[embedding_matrix], input_length=max_length, trainable=True)
    model.add(embedding_layer)
    model.add(LSTM(100))
    model.add(Dropout(0.65)) # Dropout layer
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print model.summary()

    model.fit(X_train, y_train, nb_epoch=10, batch_size=64,verbose=1, validation_data=(X_dev,y_dev), shuffle =True, callbacks=[tensorboard])
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=1)
    print "Accuracy: %.2f%%" % (scores[1]*100)
    model.save('gt_embedding_model_with_dropout.h5')  # creates a HDF5 file
