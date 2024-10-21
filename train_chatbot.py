# October 20, 2024
# required modules

import numpy as np
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Activation, Dropout
from tensorflow.python.keras.optimizer_v1 import SGD
import random

import nltk 
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle

# file to store data to train chatbot
intents_file = open('intents.json').read()
intents = json.loads(intents_file)

# tokenizing to preprocess data
words = []
classes = []
documents = []
ignore_letters = ['!', '?', '.', ',']

for intent in intents['intents']: 
    for pattern in intent['patterns']: 
        # tokenize each word
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        # add documents
        documents.append((word, intent['tag']))
        # add to class list
        if intent['tag'] not in classes: 
            classes.append(intent['tag'])
            
print(documents)

# lemmatize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

# sort classes
classes = sorted(list(set(classes)))

# documents = combination of patterns and intents
print(len(documents), "documents")

# classes = intents
print(len(classes), "classes", classes)

# words = all unique lemmatized words, vocabulary
print(len(words), "words", words)

pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

# training data
training = []

# create empty array for the output
output_empty = [0] * len(classes)

# initializing set, bag of words for every sentence
for doc in documents: 
    # initializing bag of words
    bag = []
    # list of tokenized words for pattern
    word_patterns = doc[0]
    # lemmatize each word - create base word, to attempt to represent related words
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    
    # create bag of words array with 1, if word is in current pattern
    for word in words: 
        bag.append(1) if word in word_patterns else bag.append(0)
        
    # output is 0 for each tag and 1 for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])
    
# shuffle the features and make numpy array
random.shuffle(training)
training = np.array(training)

# create training and testing lists. X - patterns, Y - intents
train_x = list(training[:,0]) 
train_y = list(training[:,1])
print('Training data is created')

# training the module
# 3 layers, 128 - 64 - len(classes) of neurons for each layer

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# avoid overfitting
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# compiling model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# training and saving model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('model created')
