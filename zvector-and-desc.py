import nltk as nltk
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
import re
import string
from nltk import word_tokenize
from sklearn.manifold import TSNE
from fastdtw import fastdtw
import multiprocessing as mp
from scipy.spatial.distance import squareform, euclidean
from functools import partial


class word2vec():

    def __init__(self):
        self.n = settings['n']
        self.eta = settings['learning_rate']
        self.epochs = settings['epochs']
        self.window = settings['window_size']

    def generate_training_data(self, settings, corpus):
        # Find unique word counts using dictonary
        word_counts = defaultdict(int)
        for sent in corpus:
            for row in sent:
                for word in row:
                    word_counts[word] += 1

            self.v_count = len(word_counts.keys())

            self.words_list = list(word_counts.keys())

            self.word_index = dict((word, i) for i, word in enumerate(self.words_list))

            self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

            training_data = []

            # Cycle through each sentence in corpus
            for sentence in sent:
                sent_len = len(sentence)

                # Cycle through each word in sentence
                for i, word in enumerate(sentence):
                    # Convert target word to one-hot
                    # print(word)
                    w_target = self.word2onehot(sentence[i])

                    # Cycle through context window
                    w_context = []

                    # Note: window_size 2 will have range of 5 values
                    for j in range(i - self.window, i + self.window + 1):
                        # Criteria for context word
                        # 1. Target word cannot be context word (j != i)
                        # 2. Index must be greater or equal than 0 (j >= 0) - if not list index out of range
                        # 3. Index must be less or equal than length of sentence (j <= sent_len-1) - if not list index out of range
                        if j != i and j <= sent_len - 1 and j >= 0:
                            # Append the one-hot representation of word to w_context
                            w_context.append(self.word2onehot(sentence[j]))


                    # training_data contains a one-hot representation of the target word and context words
                    #################################################################################################
                    # Example:																						#
                    # [Target] natural, [Context] language, [Context] processing									#
                    # print(training_data)																			#
                    # [[[1, 0, 0, 0, 0, 0, 0, 0, 0], [[0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0]]]]	#
                    #################################################################################################
                    training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]

        # Get ID of word from word_index
        word_index = self.word_index[word]

        # Change value from 0 to 1 according to ID of the word
        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):
        rgen = np.random.RandomState(1)
        self.w1 = rgen.normal(loc=0.0, scale=0.1, size=np.shape(self.n * self.n - 1))
        self.w2 = rgen.normal(loc=0.0, scale=0.1, size=np.shape(self.n * self.n - 1))


        # self.w1 = np.random.uniform(-1, 1, (self.v_count, self.n))
        # self.w2 = np.random.uniform(-1, 1, (self.n, self.v_count))

        # Cycle through each epoch
        self._errors = []

        for i in range(self.epochs):
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred, h, u = self.forward(w_t)
                EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(EI, h, w_t)
                self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
            self._errors.append(self.loss)
            print('Epoch:', i, "Loss:", self.loss)


        # cores = 10
        # batch = int(training_data.shape[0] / cores)
        # epok = [slaveId for slaveId in range(cores)]
        # #
        # for i in range(self.epochs):
        #     try:
        #         with mp.Pool(processes=cores) as pool:
        #             result = pool.starmap_async(partial(fastdtw, dist=euclidean), [self.calculate(training_data, batch, slaveId) for slaveId in epok])
        #
        #     except:
        #         print('------------------------ FINISH ------------------------------')

    # def calculate(self, training_data, batch, slaveId):
    #     self.loss = 0
    #     print('Slave id: '+str(slaveId))
    #     for w_t, w_c in training_data[(slaveId+1)*batch-batch:(slaveId+1)*batch]:
    #         y_pred, h, u = self.forward(w_t)
    #         EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
    #         self.backprop(EI, h, w_t)
    #         self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
    #     self._errors.append(self.loss)
    #     # print('Epoch:', i, "Loss:", self.loss)
    #     print('SlaveId end: '+str(slaveId))
    #     return [1, 2]



    # def parallel_for_loop(self, training_data):
        # for i in range(self.epochs):
        #     # Intialise loss to 0
        #     self.loss = 0
        #     # Cycle through each training sample
        #     # w_t = vector for target word, w_c = vectors for context words
        #     for w_t, w_c in training_data:
        #         # Forward pass
        #         # 1. predicted y using softmax (y_pred) 2. matrix of hidden layer (h) 3. output layer before softmax (u)
        #         y_pred, h, u = self.forward(w_t)
        #
        #         # Calculate error
        #         # 1. For a target word, calculate difference between y_pred and each of the context words
        #         # 2. Sum up the differences using np.sum to give us the error for this particular target word
        #         EI = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
        #
        #         # Backpropagation
        #         # We use SGD to backpropagate errors - calculate loss on the output layer
        #         self.backprop(EI, h, w_t)
        #         #########################################
        #         # print("W1-after backprop", self.w1)	#
        #         # print("W2-after backprop", self.w2)	#
        #         #########################################
        #
        #         # Calculate loss
        #         # There are 2 parts to the loss function
        #         # Part 1: -ve sum of all the output +
        #         # Part 2: length of context words * log of sum for all elements (exponential-ed) in the output layer before softmax (u)
        #         # Note: word.index(1) returns the index in the context word vector with value 1
        #         # Note: u[word.index(1)] returns the value of the output layer before softmax
        #         self.loss += -np.sum([u[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(u)))
        #     self._errors.append(self.loss)
        #     print('Epoch:', i, "Loss:", self.loss)

    def forward(self, x):
        # x is one-hot vector for target word, shape - 9x1
        # Run through first matrix (w1) to get hidden layer - 10x9 dot 9x1 gives us 10x1
        h = np.dot(x, self.w1)
        # Dot product hidden layer with second matrix (w2) - 9x10 dot 10x1 gives us 9x1
        u = np.dot(h, self.w2)
        # Run 1x9 through softmax to force each element to range of [0, 1] - 1x8
        y_c = self.softmax(u)
        return y_c, h, u

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, e, h, x):
        # https://docs.scipy.org/doc/numpy-1.15.1/reference/generated/numpy.outer.html
        # Column vector EI represents row-wise sum of prediction errors across each context word for the current center word
        # Going backwards, we need to take derivative of E with respect of w2
        # h - shape 10x1, e - shape 9x1, dl_dw2 - shape 10x9
        # x - shape 9x1, w2 - 10x9, e.T - 9x1
        dl_dw2 = np.outer(h, e)
        dl_dw1 = np.outer(x, np.dot(self.w2, e.T))
        ########################################
        # print('Delta for w2', dl_dw2)			#
        # print('Hidden layer', h)				#
        # print('np.dot', np.dot(self.w2, e.T))	#
        # print('Delta for w1', dl_dw1)			#
        #########################################

        # Update weights
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

    # Get vector from word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    # Input vector, returns nearest word(s)
    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            # Find the similary score for each word in vocab
            v_w2 = self.w1[i]
            theta_sum = np.dot(v_w1, v_w2)
            theta_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            theta = theta_sum / theta_den

            word = self.index_word[i]
            word_sim[word] = theta

        words_sorted = sorted(word_sim.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:top_n]:
            print(word, sim)


#####################################################################

# text1 = "natural language processing and machine learning is fun and exciting"
# text2 = "THe earth stupidly revolves around the sun. The moon revolves around the earth. Natural language processing and machine learning is fun and exciting"
# textList = [text1, text2]
# preprocessedTextList = []
# textLength = 0
# mergedList = ""

# for sentences in textList:
#     sentenceList = sentences.split(".")
#     if len(sentenceList) > 1:
#         for sentence in sentenceList:
#             mergedList += sentence
#         sentenceList = [mergedList]
#     corpus = [[word.split() for word in sentenceList]]
#     corpus = [[word.lower() for word in np.array(corpus).ravel()]]
#     preprocessedTextList.append(corpus)
#     textLength += len(sentences.split())

# textLength = textLength/len(textList)
# print(textLength)
# print(preprocessedTextList)


##-----------------------------------------------------------------------------------
# # Initialise object
# w2v = word2vec()
#
# # Numpy ndarray with one-hot representation for [target_word, context_words]
# training_data = w2v.generate_training_data(settings, corpus)
#
# w2v.train(training_data)
#
# # Get vector for word
# word = "beetle"
# vec = w2v.word_vec(word)
# print(word, vec)
#
# # Find similar words
# w2v.vec_sim("beetle", 3)

# training_data = w2v.generate_training_data(settings, preprocessedTextList)
# print(training_data)
# w2v.train(training_data)
#
# word = "language"
# vec = w2v.word_vec(word)
# print(word, vec)
#
# # Find similar words
# w2v.vec_sim("language", 3)



def load_dataset(filepath, cols):
    df = pd.read_csv(filepath, encoding='latin-1')
    df.columns = cols
    return df

def delete_unwanted_cols(df, cols):
    for col in cols:
        del df[col]
    return df

def preprocess_tweet_text(tweet):
    # text to lowercase
    tweet = tweet.lower()

    # remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", "", tweet, flags=re.MULTILINE)

    # remove punctuations
    tweet = tweet.translate(str.maketrans("", "", string.punctuation))

    # remove @ references and #
    tweet = re.sub(r'\@\w+|\#', "", tweet)

    # remove stopwords
    tweet_tokens = word_tokenize(tweet)
    filtered_words = [word for word in tweet_tokens if word not in stop_words]

    # stemming
    # ps = PorterStemmer()
    # stemmed_words = [ps.stem(w) for w in filtered_words]

    # lemmatizing
    # lemmatizer = WordNetLemmatizer()
    # lemma_words = [lemmatizer.lemmatize(w, pos='a') for w in stemmed_words]

    return " ".join(filtered_words)


def word_similarity_scatter_plot(index_to_word, weight):
    labels = []
    tokens = []

    for key, value in index_to_word.items():
        tokens.append(weight[key])
        labels.append(value)

    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    for i in range(len(x)):
        plt.scatter(x[i], y[i])
        plt.annotate(labels[i],
                      xy=(x[i], y[i]),
                      xytext=(5, 2),
                      textcoords='offset points',
                      ha='right',
                      va='bottom')
    plt.show()


if __name__ == '__main__':

    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))

    # Loading datasets
    columns = ['polarity', 'id', 'timestamp', 'query', 'user', 'tweet']
    dataset = load_dataset('training.1600000.processed.noemoticon.csv', columns)
    print("Collected")

    # Getting only 6000 tweets from whole data
    top = dataset.head(25)
    bottom = dataset.tail(25)
    train_dataset = pd.concat([top, bottom])
    train_dataset.reset_index(inplace=True, drop=True)

    # deleting redundant columns
    del_cols = ['timestamp', 'query', 'user']
    train_dataset = delete_unwanted_cols(train_dataset, del_cols)

    not_shuffled_d = []
    not_shuffled_l = []
    for i in range(len(train_dataset)):
        not_shuffled_d.append(train_dataset['tweet'][i])
        if train_dataset['polarity'][i] == 4:
            not_shuffled_l.append(1)
        else:
            not_shuffled_l.append(0)

    not_shuffled_d_preprocessed = []
    for i in range(len(not_shuffled_d)):
        not_shuffled_d_preprocessed.append(preprocess_tweet_text(not_shuffled_d[i]))

    tweets = []
    for i in range(len(not_shuffled_d)):
        tweets.append(not_shuffled_d_preprocessed[i])

    text = ""
    for tweet in tweets:
        text += tweet

    textList = [text]

    mergedList = ""
    preprocessedTextList = []
    textLength = 0

    for sentences in textList:
        sentenceList = sentences.split(".")
        if len(sentenceList) > 1:
            for sentence in sentenceList:
                mergedList += sentence
            sentenceList = [mergedList]
        corpus = [[word.split() for word in sentenceList]]
        corpus = [[word.lower() for word in np.array(corpus).ravel()]]
        preprocessedTextList.append(corpus)
        textLength += len(sentences.split())

    settings = {
        'window_size': 2,  # context window +- center word
        'n': 20,  # dimensions of word embeddings, also refer to size of hidden layer
        'epochs': 300,  # number of training epochs
        'learning_rate': 0.01  # learning rate
    }
    w2v = word2vec()

    training_data2 = w2v.generate_training_data(settings, preprocessedTextList)

    w2v.train(training_data2)
    word3 = "facebook"
    vec = w2v.word_vec(word3)
    w2v.vec_sim("facebook", 6)
    word_similarity_scatter_plot(w2v.index_word, w2v.w1)



