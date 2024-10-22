### Graphical interface for chatbot
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.python.keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))


# clean_up_sentences takes a sentence and returns a bag of words array: 0 or 1 for words 
# that exist in sentence
# clean_up_sentences: str -> List(str)
def clean_up_sentences(sentence):
    # tokenize sentence - split into array
    sentence_words = nltk.word_tokenize(sentence)
    # stemming every word - reduce to base form
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# bag_of_words takes sentence, words and returns bag of words - vocab matrix
# bag_of_words: str, str -> np.array
def bag_of_words(sentence, words, show_details=True): 
    # tokenizing patterns
    sentence_words = clean_up_sentences(sentence)
    # bag of words - vocab matrix
    bag = [0] * len(words)
    for s in sentence_words: 
        for i,word in enumerate(words): 
            if word == s: 
                # assign 1 if current word is in vocab position
                bag[i] = 1
                if show_details: 
                    print('found in bag: %s' % word)
    return(np.array(bag))

# predict_class returns the probably of the class sentence belongs to 
# predict_class: str -> List(Dict[str: num])
def predict_class(sentence):
    ERROR_THRESHOLD = 0.25
    return_list = []
    
    # filter threshold predictions
    p = bag_of_words(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    results = [[i,r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    
    # sort by probability strength
    results.sort(key=lambda x:x[1], reverse=True)
    for r in results: 
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

# getResponse produces a response 
# getResponse: List(Dict[str: str]), List(Dict[str: str])
def getResponse(ints, intent_json): 
    tag = ints[0]['intent']
    list_of_intents = intent_json['intents']
    
    for i in list_of_intents: 
        if (i['tag'] == tag): 
            result = random.choice(i['responses'])
            break
    
    return result

# creating tkinter GUi
import tkinter as tk
from tkinter import *

def send(): 
    msg = tk.Entry.get("1.0", "end-1c").strip()
    Entry.delete('0.0', END)
    
    if msg != '': 
        ChatBox.config(state=NORMAL)
        ChatBox.insert(END, "You: " + msg + '\n\n')
        ChatBox.config(foreground='#446665', font=("Verdana", 12))
        
        ints = predict_class(msg)
        res = getResponse(ints, intents)
        
        ChatBox.insert(END, "Bot: " + res + '\n\n')
        
        ChatBox.config(state=DISABLED)
        ChatBox.yview(END)
        
root = Tk()
root.title("ChatBot")
root.geometry("400x500")
root.resizable(width=False, height=False)

# Create chat box window
ChatBox = Text(root, bd=0, bg="white", height="8", width="50", font="Arial",)
ChatBox.config(state=DISABLED)

# Bind scrollbar to Chat window
scrollbar = Scrollbar(root, command=ChatBox.yview, cursor="heart")
ChatBox['yscrollcommand'] = scrollbar.set

#Create Button to send message
SendButton = Button(root, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
                    bd=0, bg="#f9a602", activebackground="#3c9d9b",fg='#000000',
                    command= send )

#Create the box to enter message
EntryBox = Text(root, bd=0, bg="white",width="29", height="5", font="Arial")

#EntryBox.bind("<Return>", send)
#Place all components on the screen
scrollbar.place(x=376,y=6, height=386)
ChatBox.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=128, y=401, height=90, width=265)
SendButton.place(x=6, y=401, height=90)

root.mainloop()
