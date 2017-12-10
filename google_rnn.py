
import json
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils




if __name__ == '__main__':
    # Open up pre-processed location corpus
    with open('resources/english_sms_corpus.json', 'r') as fp:
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


    #get single words out of sentences
    tokMessages = []

    maxlength = 0
    longest = []
    for msg in messages:
        tok = word_tokenize(msg)
        tokMessages.append(tok)
        if len(tok) > maxlength:
            maxlength =len(tok)
            longest = tok

    newMsg =[]
    for msg in tokMessages:
        n = " ".join(msg)
        newMsg.append(n)

    toker = Tokenizer(filters = '')
    toker.fit_on_texts(newMsg)
    X_train = toker.texts_to_sequences(newMsg)
    print(len(toker.word_counts))
    X_train = np.asarray(X_train)

    print (np.shape(tokMessages))
    print(maxlength)
    #print(" ".join (longest))

    print(np.shape(locations))
    print(locations[2])
    y_train = locations
    y_train[locations == 'sg'] = 0
    y_train[locations == 'us'] = 1
    y_train[locations == 'uk'] = 2

    # Get "location" embeddings
    #word_to_index = toker.word_index

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

    vocabulary = toker.word_index.keys()
    for word in vocabulary:
        if word in vocab_weights:
            index = toker.word_index[word]
            embedding_matrix[index] = vocab_weights[word]


    embedding_matrix = np.array(embedding_matrix)
    print "EMBEDDING MATRIX"
    print embedding_matrix
    print np.shape(embedding_matrix)


# LSTM for sequence classification in the IMDB dataset
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
np.random.seed(43)
# load the dataset but only keep the top n words, zero the rest
#top_words = 5000


#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=top_words)
#print(X_train[1])

num_classes = []
# truncate and pad input sequences
max_length = 200
X_train = sequence.pad_sequences(X_train, maxlen=max_length)
y_train = np_utils.to_categorical(y_train, num_classes)
print(y_train)
# create the model
#embedding_vector_length = 32
embedding_vector_length = 3
model = Sequential()


#model.add(Embedding(len(toker.word_counts)+1, embedding_vector_length, input_length=max_length))
embedding_layer = Embedding(len(toker.word_counts)+1, embedding_vector_length, weights=[embedding_matrix], input_length=max_length, trainable=True)
model.add(embedding_layer)
model.add(LSTM(100))
#Dropout
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, nb_epoch=10, batch_size=64,verbose=1, validation_split=0.1, shuffle =True)
# Final evaluation of the model
#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Accuracy: %.2f%%" % (scores[1]*100))
