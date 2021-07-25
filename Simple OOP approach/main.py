from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from preprocessing import Preprocessing, Sampling
from tensorflow.keras.models import load_model
import numpy as np
import random
import fitz
import sys

x, y, seq_length, n_vocab, sentence, int_to_char, char_to_int, raw_text, n_chars = Preprocessing.preprocess(
    "PATH_TO_CORPUS")

# filepath = "best_model-{epoch:02d}.hdf5"
# checkpoint = ModelCheckpoint(filepath,
#                              monitor='loss',
#                              verbose=1,
#                              save_best_only=True,
#                              mode='min')
# callbacks_list = [checkpoint]


model = Sequential([
    LSTM(256, input_shape=(seq_length, n_vocab),  return_sequences=True),
    Dropout(0.1),
    LSTM(256),
    Dense(n_vocab, activation='softmax')
])
optimizer = Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer)

# model.load_weights("best_model_2-05.hdf5") # if weights are saved or else train

history = model.fit(x, y,
                    # Be careful with the batch size - lower it if hits memory issue.
                    batch_size=32,
                    epochs=20
                    )


np.seterr(divide='ignore')  # ignore the warning of divide by zero.

start_index = random.randint(0, n_chars - seq_length - 1)
generated = ''
# Getting a random sentence from the corpus.
sentence = raw_text[start_index: start_index + seq_length]
generated += sentence
print('Input sequence:"' + sentence + '"\n')

for i in range(600):   # Number of characters including spaces
    x_pred = np.zeros((1, seq_length, n_vocab))
    for t, char in enumerate(sentence):
        # Preparing the x we want to predict as we have done for training. The full sentence is in shape of (1, 60, 69)
        x_pred[0, t, char_to_int[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = Sampling.sample(preds)  # Providing stochasticty
    # next_index = np.random.choice(y.shape[1], 1, p=preds)[0] # Another way to choose index with stochasticty and providing the probability distribution of the preds - which is very important
    next_char = int_to_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char  # move the window by 1 character

    sys.stdout.write(next_char)
    sys.stdout.flush()
print()
