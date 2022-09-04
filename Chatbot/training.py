#Import libraries
import random                                                   #Generate random value
import json                                                     #Javascript object notation
import pickle                                                   #For converting python object
import numpy as np                                              #Muldimensional array library
import nltk                                                     #Nature language toolkit   
from nltk.stem import WordNetLemmatizer                         #Lemmatizer
from rsa import verify
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()
#Lemmatizer is a process of grouping together the different inflected forms of a word so they can be analyzed as a single item.

intents = json.loads(open('intents.json').read())   #Load json file

#create empty list
words = []
classes = []
documents = []
ignore_letters = ['?','!','.',',']  

#intent accessing
for intent in intents['intents']:                       #accessing intents
    for pattern in intent['patterns']:                  #accessing patterns in intents 
        word_list = nltk.word_tokenize(pattern)         #tokenize is splitting up a larger body of text into smaller lines
        words.extend(word_list)                         #adds all the elements of an iterable (list, tuple, string etc.) to the end of the list.
        documents.append((word_list, intent['tag']))    #append word list with tag
        if intent['tag'] not in classes:                #check the tag is in the classes
            classes.append(intent['tag'])

#word lemmatizer
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]

#eliminate duplicate
words = sorted(set(words))  
classes = sorted(set(classes))

#file saving (wb =writing binary)
pickle.dump(words, open('words.pkl', 'wb'))     
pickle.dump(classes, open('classes.pkl', 'wb'))

#Machine Learning part
#Neuron network need numerical values, so we gonna represent all words above with numerical values with bag of words method
training = []
output_empty = [0] * len(classes)

#bag of words
for document in documents:
    bag = []    #empty bag for different combination
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    #checking each word occur a patterns
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0) #'1' if word occuring a pattern otherwise '0'
        
        #checking for output row (training list)
        output_row = list(output_empty)
        output_row[classes.index(document[1])] = 1
        training.append([bag, output_row])  #append the bag with training list

random.shuffle(training)            #shuffle the training data
training = np.array(training)       #make it to numpy array

#splitting x and y values
train_x = list(training[:, 0])
train_y = list(training[:, 1])

model = Sequential()                #create sequential model
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))    #neuron layers
model.add(Dropout(0.5))                                                     #avoid overfitting
model.add(Dense(64, activation='relu'))                                     #Dense layer
model.add(Dropout(0.5))                                                     #avoid overfitting
model.add(Dense(len(train_y[0]), activation='softmax'))                     #sums up input and output layer

#Statistic Gradient Descent is the optimizer
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)                         
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy']) 

#compiling the model to chatbotmodel
hist = model.fit(np.array(train_x), np.array(train_y), epochs=100, batch_size=5, verbose=1)
model.save('chatbotmodel.h5', hist)

print("Done")