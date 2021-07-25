import numpy as np
import random
import fitz


class Preprocessing():
    @staticmethod
    def preprocess(path):
        with fitz.open(path) as doc:
            text = str()
            for page in doc:
                text += page.getText()
        raw_text = text

        # it's a long preprocessed string.
        raw_text = "".join(c for c in raw_text if c.isascii() and not c.isdigit())
        # Taking only the unique characters with other special characters also.
        chars = sorted(list(set(raw_text)))

        char_to_int = dict((c, i)for i, c in enumerate(chars))
        int_to_char = {v: k for k, v in char_to_int.items()}

        n_chars = len(raw_text)
        n_vocab = len(chars)

        seq_length = 60  # 60 charachter as input to  model.
        step = 1  # How far we will take the steps.
        sentences = []  # x
        next_chars = []  # y

        # Creating list of sentences and next_chars (x and y)
        for i in range(0, len(raw_text) - seq_length, step):
            sentences.append(raw_text[i: i+seq_length])
            next_chars.append(raw_text[i+seq_length])

        # Preparing the text to be sutiable as an input to the model (matrix representation).
        x = np.zeros((len(sentences), seq_length, n_vocab), dtype=np.bool)
        y = np.zeros((len(sentences), n_vocab), dtype=np.bool)

        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                x[i, t, char_to_int[char]] = 1
            y[i, char_to_int[next_chars[i]]] = 1

        return x, y, seq_length, n_vocab, sentence, int_to_char, char_to_int, raw_text, n_chars


class Sampling():
    @staticmethod
    def sample(preds):
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds)
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)
