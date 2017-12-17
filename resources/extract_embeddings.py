import json
import numpy as np
#from resources.importData import importAndProcess
from time import time
import pylab as Plot
import random

from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from keras.preprocessing.text import text_to_word_sequence
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils
from keras.utils import plot_model
from keras.callbacks import TensorBoard
from keras.models import load_model

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

from keras import backend as bk

if __name__ == '__main__':
    # Get words and indices
    word_indices = {}
    with open('word_indicies.txt', 'r') as fp:
        wi_lines = fp.readlines()
        for line in wi_lines:
            line = line.strip().split(',')
            word_indices[int(line[1])] = line[0]

    model = load_model('../gt_embedding_model.h5')
    weights = model.layers[0].get_weights()[0]

    sg_words = ['lah', 'handphone', 'hdb', 'shuhui', 'chio', 'pinyin']
    uk_words = ['flat', 'colour', 'realised', 'lorry', 'sacked', 'mum', 'pub', 'pubs', 'learnt', 'film', 'centre', 'mobile', 'marks']
    us_words = ['apartment', 'color', 'realized', 'truck', 'fired', 'mom', 'movie', 'center', 'grade']

    other = []
    #random.seed(19)
    #random.seed(16)
    #random.seed(13)
    #random.seed(10)
    random.seed(47)
    for i in range(0, 50):
        other.append(random.choice(list(word_indices.values())))

    words_of_interest = sg_words + uk_words + us_words + other
    to_plot_matrix = []

    for word in words_of_interest:
        found = False
        for key, value in word_indices.items():
            if value == word:
                found = True
                #indices_of_interest.append(key)
                word_weight = weights[key]
                to_plot_matrix.append(word_weight)
                break
        if found == False:
            print "We didn't find \'{}\' in the corpus.".format(word)

    to_plot_matrix = np.array(to_plot_matrix)


    tsne_model = TSNE(n_components=2, random_state=43)
    vectors = tsne_model.fit_transform(to_plot_matrix)

    #normalizer = preprocessing.Normalizer()
    #vectors = normalizer.fit_transform(vectors, 'l1')
    #print vectors

    fig, ax = plt.subplots(figsize=(20, 7))
    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')
    for i, word in enumerate(words_of_interest):
        if word in sg_words:
            marker_color = 'b'
        elif word in uk_words:
            marker_color = 'r'
        elif word in us_words:
            marker_color = 'g'
        else:
            marker_color = 'black'
        #ax.annotate(word, (vectors[i][0], vectors[i][1]), vectors[i][2])
        #point = vectors[i][0], vectors[i][1], vectors[i][2]
        #point = vectors[i][0], vectors[i][1]
        ax.scatter(vectors[i][0], vectors[i][1], s=2, color=marker_color)
        ax.text(vectors[i][0], vectors[i][1], '%s' % (word), zorder=1, color=marker_color)

    #plt.xlim(-5, 5)
    #plt.ylim(-5, 5)
    #plt.margins(0)

    plt.savefig('embeddings47.png')


    #Plot.savefig("embeddings.png");
