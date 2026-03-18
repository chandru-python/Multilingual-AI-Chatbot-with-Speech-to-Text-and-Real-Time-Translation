import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
from duckduckgo_search import DDGS  

# Initialize lemmatizer and load model and data
lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('new.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

ERROR_THRESHOLD = 0.5

def clean_up_sentence(sentence):
    # Tokenize and lemmatize words
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words):
    # Create a bag of words array
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    # Predict class based on sentence
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if not results:
        return None  # No high-confidence intent found
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    if ints is None:
        return None  # Signal that no intent was found

    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    
    return None  # Default case (should not be reached)

def search_web(query):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=2))  
        if results:
            return "\n".join([r["body"] for r in results])  
        return "I couldn't find relevant information."
    except Exception:
        return "I couldn't fetch web results at the moment."

def chatbot_response(msg):
    ints = predict_class(msg, model)
    response = getResponse(ints, intents)
    
    if response is None:  # No matching intent
        return search_web(msg)
    
    return response

# Sample interaction (console-based)
if __name__ == "__main__":
    print("Hello! I'm your chatbot. Type 'quit' to end.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break
        response = chatbot_response(user_input)
        print(f"Bot: {response}")
