import random                               #Generate random value
import json                                 #Javascript object notation
import pickle                               #For converting python object
import numpy as np                          #Muldimensional array library
import nltk                                 #Nature language toolkit    
from nltk.stem import WordNetLemmatizer     #Lemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer() 
#Lemmatizer is a process of grouping together the different inflected forms of a word so they can be analyzed as a single item.                    
intents = json.loads(open('intents.json').read())   #load json file

#Opening the file in rb or reading binary mode
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

#cleaning the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#convert sentence to bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#predicting class base on sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0] #predict result base bag of words
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD] #enumerate and prevent uncertainty by error_treshold

    #sort result by probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list
#   
def get_response(intents_list, intent_json):
    tag = intents_list[0]['intent']
    list_of_intents = intent_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result
print("\nThis is IEEE HKN MU Alpha Chapter chatbot developed by Nico Halisno the Vice President of IEEE")
print("Take a note this AI chatbot still under development")
print("\nThe AI chatbot can let the user ask any questions about IEEE HKN")
print("Feel free to ask any questions and don't bully the chatbot yee :D \n")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
