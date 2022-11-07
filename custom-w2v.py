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


class word2vec():

    def __init__(self, epochs, window, n=20, eta=0.01):
        self.n = n
        self.eta = eta
        self.epochs = epochs
        self.window = window

    def generate_training_data(self, corpus):
        word_counts = defaultdict(int)
        for sent in corpus: #[[], [], [], [], []][[], [], [], [], []] - matrix
            for row in sent: #[[],[],[]] - vector
                for word in row: #['sadf', 'sdf', 'sdf']
                    word_counts[word] += 1

            self.v_count = len(word_counts.keys())

            self.words_list = list(word_counts.keys())

            self.word_index = dict((word, i) for i, word in enumerate(self.words_list))

            self.index_word = dict((i, word) for i, word in enumerate(self.words_list))

            training_data = []
            for sentence in sent:
                sent_len = len(sentence)
                for i, word in enumerate(sentence):
                    w_target = self.word2onehot(sentence[i])
                    w_context = []
                    for j in range(i - self.window, i + self.window + 1):
                        if j != i and sent_len - 1 >= j >= 0:
                            w_context.append(self.word2onehot(sentence[j]))

                    training_data.append([w_target, w_context])

        return np.array(training_data)

    def word2onehot(self, word):
        word_vec = [0 for i in range(0, self.v_count)]

        word_index = self.word_index[word]
        word_vec[word_index] = 1

        return word_vec

    def train(self, training_data):
        rgen = np.random.RandomState(1)
        self.w1 = rgen.normal(loc=0.0, scale=0.1, size=np.shape(self.n * self.n - 1))
        self.w2 = rgen.normal(loc=0.0, scale=0.1, size=np.shape(self.n * self.n - 1))
        for i in range(self.epochs):
            self.loss = 0
            for w_t, w_c in training_data:
                y_pred, h, out = self.forward(w_t)
                error = np.sum([np.subtract(y_pred, word) for word in w_c], axis=0)
                self.backprop(error, h, w_t)
                self.loss += -np.sum([out[word.index(1)] for word in w_c]) + len(w_c) * np.log(np.sum(np.exp(out)))
            print('Epoch:', i, "Loss:", self.loss)


    def forward(self, x):
        h = np.dot(x, self.w1)
        out = np.dot(h, self.w2)
        y_c = self.softmax(out)
        return y_c, h, out

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)

    def backprop(self, error, h, x):
        dl_dw2 = np.outer(h, error)
        dl_dw1 = np.outer(x, np.dot(self.w2, error.T))
        self.w1 = self.w1 - (self.eta * dl_dw1)
        self.w2 = self.w2 - (self.eta * dl_dw2)

    # Get vector from word
    def word_vec(self, word):
        w_index = self.word_index[word]
        v_w = self.w1[w_index]
        return v_w

    def vec_sim(self, word, top_n):
        v_w1 = self.word_vec(word)
        word_sim = {}

        for i in range(self.v_count):
            v_w2 = self.w1[i]
            t_sum = np.dot(v_w1, v_w2)
            t_den = np.linalg.norm(v_w1) * np.linalg.norm(v_w2)
            t = t_sum / t_den
            word = self.index_word[i]
            word_sim[word] = t

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

    w2v = word2vec(n=20, eta=0.01, epochs=300, window=2)

    training_data2 = w2v.generate_training_data(preprocessedTextList)

    w2v.train(training_data2)
    word3 = "facebook"
    vec = w2v.word_vec(word3)
    w2v.vec_sim("facebook", 11)
    word_similarity_scatter_plot(w2v.index_word, w2v.w1)



