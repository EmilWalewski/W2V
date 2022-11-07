import string
from nltk import word_tokenize
from nltk.corpus import stopwords
from gensim.models import Word2Vec
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
    tweets.append(not_shuffled_d_preprocessed[i].split())



model = Word2Vec(tweets, min_count=1,workers=1)
# model.save("word2vec.model")

# model = Word2Vec.load("word2vec.model")
model.train(tweets, total_examples=50, epochs=300)
words = list(model.wv.index_to_key)
print(model.wv.most_similar('facebook', topn=10))
# vector = model.wv['final']

vocab = list(model.wv.key_to_index)
X = model.wv[vocab]
tsne = TSNE(n_components=2)
X_tsne = tsne.fit_transform(X)
df = pd.DataFrame(X_tsne, index=vocab, columns=['x', 'y'])
fig = plt.figure(figsize=(10, 20))
ax = fig.add_subplot(1, 1, 1)
ax.scatter(df['x'], df['y'])
for word, pos in df.iterrows():
    ax.annotate(word, pos)
plt.show()