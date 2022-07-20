
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import Counter
import pickle


import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

import gensim

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


import re

# To read the Data
df = pd.read_csv('trainingdata.csv', encoding = 'latin',header=None)


# Labeling all the columns according to the data
df.columns = ['sentiments', 'id', 'date', 'query', 'userid', 'text']

# Droping all the rows which we do no need for the project
# axis=1 mean rows and axis=0 means columns
df = df.drop(['id', 'date', 'query', 'userid'], axis=1)

# Mapping target label to 0-negative 2-neutral 4-positive
decode_map = {0: "NEGATIVE", 2: "NEUTRAL", 4: "POSITIVE"}
def decode_sentiment(label):
    return decode_map[int(label)]
df.sentiments = df.sentiments.apply(lambda x : decode_sentiment(x))

# Stemming The words
stop_words = stopwords.words("english")
stemmer = SnowballStemmer("english")


text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
def preprocess(text, stem=False):
    # Remove link,user and special characters
    text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
    tokens = []
    for token in text.split():
        if token not in stop_words:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)

df.text = df.text.apply(lambda x: preprocess(x))

print(df.head())

# Splitting the Test and Train data
TRAIN_SIZE = 0.8
train_data, test_data = train_test_split(df, test_size=1-TRAIN_SIZE, random_state=7) # Splits Dataset into Training and Testing set
print("Train Data size:", len(train_data))
print("Test Data size", len(test_data))

#Word2Vec Algorithm implementation
W2V_SIZE = 30
W2V_WINDOW = 7
W2V_EPOCH = 32
W2V_MIN_COUNT = 10

documents = [_text.split() for _text in train_data.text]

w2v_model = gensim.models.word2vec.Word2Vec(vector_size=W2V_SIZE, window=W2V_WINDOW, min_count=W2V_MIN_COUNT, workers=8)
w2v_model.build_vocab(documents)
w2v_model.train(documents, total_examples=len(documents), epochs=W2V_EPOCH)

print(w2v_model.wv.most_similar("love"))

#Tokenizing

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train_data.text)

word_index = tokenizer.word_index
vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size :", vocab_size)

from keras_preprocessing.sequence import pad_sequences
x_train = pad_sequences(tokenizer.texts_to_sequences(train_data.text),maxlen = 30)
x_test = pad_sequences(tokenizer.texts_to_sequences(test_data.text), maxlen = 30)

print("Training X Shape:",x_train.shape)
print("Testing X Shape:",x_test.shape)

#Labeling the Elements in the dataset
labels = train_data.sentiments.unique().tolist()
labels.append("NEUTRAL")
print(labels)

#Label Encoding
encoder = LabelEncoder()
encoder.fit(train_data.sentiments.to_list())

y_train = encoder.transform(train_data.sentiments.to_list())
y_test = encoder.transform(test_data.sentiments.to_list())

y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#Embedding
embedding_matrix = np.zeros((vocab_size, W2V_SIZE))
for word, i in tokenizer.word_index.items():
  if word in w2v_model.wv:
    embedding_matrix[i] = w2v_model.wv[word]
print(embedding_matrix.shape)

embedding_layer = tf.keras.layers.Embedding(vocab_size, W2V_SIZE, weights=[embedding_matrix], input_length=30, trainable=False)

model = tf.keras.models.Sequential()
model.add(embedding_layer)
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()


model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])

callbacks = [ tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=5, cooldown=0),
              tf.keras.callbacks.EarlyStopping(monitor='val_acc', min_delta=1e-4, patience=5)]

SEQUENCE_LENGTH = 30
EPOCHS = 8
BATCH_SIZE = 1024
history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1, verbose=1, callbacks=callbacks)


score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])

SENTIMENT_THRESHOLDS = (0.4, 0.7)
#Predict
def decode_sentiment(score, include_neutral=True):
    if include_neutral:
        label = "NEUTRAL"
        if score <= SENTIMENT_THRESHOLDS[0]:
            label = "NEGATIVE"
        elif score >= SENTIMENT_THRESHOLDS[1]:
            label = "POSITIVE"
        return label
    else:
        return "NEGATIVE" if score < 0.5 else "POSITIVE"


def predict(text, include_neutral=True):
    # Tokenize text
    x_test = pad_sequences(tokenizer.texts_to_sequences([text]), maxlen=SEQUENCE_LENGTH)
    # Predict
    score = model.predict([x_test])[0]
    # Decode sentiment
    label = decode_sentiment(score, include_neutral=include_neutral)

    return {"label": label, "score": float(score)}


print(predict("I love the music"))
print(predict("I hate the music"))



# EXPORT
KERAS_MODEL = "model.h5"
WORD2VEC_MODEL = "model.w2v"
TOKENIZER_MODEL = "tokenizer.pkl"
ENCODER_MODEL = "encoder.pkl"
model.save(KERAS_MODEL)
w2v_model.save(WORD2VEC_MODEL)
pickle.dump(tokenizer, open(TOKENIZER_MODEL, "wb"), protocol=0)
pickle.dump(encoder, open(ENCODER_MODEL, "wb"), protocol=0)

















