# Generating text using LSTM (character-level).

In this repository I will present how to generate text using LSTM in character level. 

# Dataset and preprocessing:

A novel (Crime and Punishment) in pdf format. The dataset that I have used is a novel and it is in pdf format, extracted and preprocessed taking only ASCII characters and removing numbers. 


# Data properties

Each training sample consists of 60 characters, with the next characters as the label. In brief, it is like a sliding window of size 60 that covers the entire corpus (except the last 60 characters).
Each sentence is converted to a 2D matrix, and each label is converted to a 1D matrix. Based on my dataset, the total number of unique characters is 69. The total number of sentences is 1152927, and each sentence is encoded in a 60 by 69 matrix. Each label is encoded as a 1D matrix of size 69. 
# Network architecture

```python
 model = Sequential([
    LSTM(256, input_shape=(seq_length, n_vocab),  return_sequences=True),
    Dropout(0.1),
    LSTM(256),
    Dense(n_vocab, activation='softmax')
])
```


# Training

The whole corpus is used to train the network. The total number of the training sample is 1152927.


# Testing

Tested with a randomly chosen subset of string from the corpus of length 60 characters. Before passing the test data to the model, it has to be pre-processed accordingly to the model expectations.

Another interesting fact is not to select the character that the model sets with high probability, since this will lead to the text generated that has repetitive characters. In this case, I have taken two different approaches to induce stochastic nature in selecting the next character

You can also use your own string but for now, the string must be a length of 60 character



# Challenges

LSTM is extremely slow to train even in GPU (I have used Colab for this) And trained the network (saved it and trained again from where it left) in two sessions for 10 epochs. The batch size needs to be carefully set to overcome memory issues. One is using a function and another is randomly choosing from the output of the model prediction but with providing the probability distribution of the model output. 

# Result 

(Network trained for 10 epochs)


```
Input Sequence : <start>d you will live. You have long needed a change of air. Suffe<end>

Output Sequence: <start>r, and said now, she go in in the 

dark on waterally. From all my honour of you go to itself." 
 
"It's all none!" 
 
"If in sleed!" cried Sonia. "It wasn't no 
account that he has been jaking 
a highly and agonies carrying out your table while meen that some witness, set all bit to the stairs ago investigion! -vestising!" 
 
Perceptical air trembling wretched waitingly in other papers. 
 
"Mich, take a kind young, By nature! What does he Katerina Ivanovna?" 
 
"I knew nothing, for I 
would know all the 
Most Razumihin?" 
 
"I don't understand, I'll go," said Raskolnikov's 
good-humoured: "/M<end>

```


---

## TODO
* [ ] Proper documentation
* [ ] Allow custom sentence as test case, without size constraint (now it's 60 character)
* [ ] Further pre-processing of the text
* [ ] Experiment with the network architecture 
* [ ] Increase training number of epochs (50?)
* [ ] Larger corpus (all Dostoyevsky's novel and short stories)

---



Reference: https://github.com/bnsreenu/python_for_microscopists/blob/master/167-LSTM_text_generation_ENGLISH.py

