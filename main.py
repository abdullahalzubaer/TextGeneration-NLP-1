import random
import sys

import fitz  # If data is from a pdf file.
import numpy as np
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential, load_model
from keras.optimizers import Adam

with fitz.open("DOCUMENT_PATH") as doc:
    text = str()
    for page in doc:
        text += page.getText()
raw_text = text


# it's a long preprocessed string.
raw_text = "".join(c for c in raw_text if c.isascii() and not c.isdigit())
# Taking only the unique characters with other special characters also.
chars = sorted(list(set(raw_text)))


char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = {v: k for k, v in char_to_int.items()}

n_chars = len(raw_text)
n_vocab = len(chars)

seq_length = 60  # 60 charachter as input to  model.
step = 1  # How far we will take the steps.
sentences = []  # x
next_chars = []  # y


# Creating list of sentences and next_chars (x and y)
for i in range(0, len(raw_text) - seq_length, step):
    sentences.append(raw_text[i : i + seq_length])
    next_chars.append(raw_text[i + seq_length])


# Preparing the text to be sutiable as an input to the model (matrix representation).
x = np.zeros((len(sentences), seq_length, n_vocab), dtype=np.bool)
y = np.zeros((len(sentences), n_vocab), dtype=np.bool)


for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_to_int[char]] = 1
    y[i, char_to_int[next_chars[i]]] = 1


# Model checkpoint

filepath = "best_model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(
    filepath, monitor="loss", verbose=1, save_best_only=True, mode="min"
)
callbacks_list = [checkpoint]

# model = load_model("/content/best_model_2-05.hdf5")


model = Sequential(
    [
        LSTM(256, input_shape=(seq_length, n_vocab), return_sequences=True),
        Dropout(0.1),
        LSTM(256),
        Dense(n_vocab, activation="softmax"),
    ]
)
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)


history = model.fit(
    x,
    y,
    # Be careful with the batch size - lower it if hits memory issue.
    batch_size=32,
    epochs=20,
    callbacks=callbacks_list,
)


# For testing the trained model.

start_index = random.randint(0, n_chars - seq_length - 1)
generated = ""
# Getting a random sentence from the corpus.
sentence = raw_text[start_index : start_index + seq_length]
generated += sentence
print('Input sequence:"' + sentence + '"\n')

# For inducing stochasticty.


def sample(preds):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


np.seterr(divide="ignore")  # ignore the warning of divide by zero.


for i in range(600):  # Number of characters including spaces
    x_pred = np.zeros((1, seq_length, n_vocab))
    for t, char in enumerate(sentence):
        # Preparing the x we want to predict as we have done for training. The full sentence is in shape of (1, 60, 69)
        x_pred[0, t, char_to_int[char]] = 1.0

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds)  # Providing stochasticty
    # next_index = np.random.choice(y.shape[1], 1, p=preds)[0] # Another way to choose index with stochasticty and providing the probability distribution of the preds - which is very important
    next_char = int_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char  # move the window by 1 character

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()
