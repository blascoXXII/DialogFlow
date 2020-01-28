import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

import numpy
import tensorflow
import tflearn
import random
import json
import pickle

with open(r"C:\Users\Blasco\Desktop\Chatbot\intents.json") as file:
    data = json.load(file)

words =[]
labels=[]
docs_x=[]
docs_y=[]

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrds = nltk.word_tokenize(pattern)  #Gets the root of the word (ignores noice) by tokenizing
        words.extend(wrds)  #add to list
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
        
    if intent["tag"] not in labels:
        labels.append(intent["tag"])
        
words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words))) #sets removes duplicates, list transfor it to list, sorted sorts the list

labels = sorted(labels)

#Neural networks only understand numbers, so in order to train our model strings have to be into lists that
#contains what characters appear in the sentence (0 if it doesn't appear 1 if it does).

training = []
output = []

out_empty = [0 for _ in range(len(labels))]
 
for x, doc in enumerate(docs_x):
    bag = []
    
    wrds = [stemmer.stem(w) for w in doc]
    
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
            
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1
    
    training.append(bag)
    output.append(output_row)
    
training = numpy.array(training)
output = numpy.array(output)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])

#we add 2 hidden layers with 8 neurons
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)


#output layer with 6 neurons. It has softmax activation, what it does is to give a probavility to each neuron.
net = tflearn.fully_connected(net,len(output[0]), activation="softmax")
net = tflearn.regression(net)

#Model trainning
model = tflearn.DNN(net)

model.fit(training, output, n_epoch=1500, batch_size=8, show_metric=True ) #n_epoch is the number of times the model will use the same data
model.save("model.tflearn")

def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
        
    return numpy.array(bag)


def chat():
    print("Start talking with the bot! (type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower()=="quit":
            break
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        print(tag, " - ",results[0][results_index])
        
        if results[0][results_index]>0.5:
        
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg["responses"]
                    
            print (random.choice(responses))
        else:
            print("Sorry, I haven't enough seniority to answer that")
        
        
chat()