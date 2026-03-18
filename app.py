from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import sqlite3
import speech_recognition as sr
from werkzeug.security import generate_password_hash, check_password_hash
import os
import joblib
import openai
from deep_translator import GoogleTranslator
from keras.models import load_model
import nltk
import numpy as np
import json
import random
import pickle
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# SQLite DB path
DB_PATH = "chatbot.db"

# Create DB and table if not exists
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            email TEXT UNIQUE,
            number TEXT,
            password TEXT,
            location TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# DB connection
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# OpenAI API key
openai.api_key = 'your-openai-api-key'

lemmatizer = WordNetLemmatizer()
model = load_model('chatbot_model.h5')
intents = json.loads(open('new.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

ERROR_THRESHOLD = 0.25

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]

def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)

    if not results:
        return None

    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def getResponse(ints, intents_json):
    if ints is None:
        return None

    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])

    return None

def chatbot_response(msg):
    ints = predict_class(msg, model)
    response = getResponse(ints, intents)

    if response is None:
        return "I couldn't understand that. Can you please rephrase?"

    return response

def translate_text(text, source_lang, target_lang):
    return GoogleTranslator(source=source_lang, target=target_lang).translate(text)

@app.route('/get-bot-response', methods=['POST'])
def get_bot_response():
    user_input = request.json.get('userInput')
    response = chatbot_response(user_input)
    return jsonify({"response": response})

@app.route('/translate', methods=['POST'])
def translate_message():
    data = request.get_json()
    text = data.get('text', '')
    source_lang = data.get('sourceLang', 'en')
    target_lang = data.get('targetLang', 'en')

    try:
        translated_text = translate_text(text, source_lang, target_lang)
        return jsonify({"translatedText": translated_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text():
    recognizer = sr.Recognizer()
    audio_file = request.files.get('audio', None)

    if not audio_file:
        return jsonify({"response": "No audio file received."}), 400

    target_language = request.form.get('targetLang', 'en')

    try:
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            text = recognizer.recognize_google(audio_data)

        if target_language != 'en':
            text = GoogleTranslator(source='auto', target='en').translate(text)

        response_text = chatbot_response(text)

        if target_language != 'en':
            response_text = GoogleTranslator(source='en', target=target_language).translate(response_text)

        return jsonify({"response": response_text})

    except Exception as e:
        return jsonify({"response": str(e)}), 500

@app.route('/chatbot')
def chatbot():
    if 'email' not in session:
        flash('Please log in to continue.', 'warning')
        return redirect(url_for('login'))
    return render_template('chatbot.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        number = request.form['number']
        password = generate_password_hash(request.form['password'])
        location = request.form['location']

        conn = get_db_connection()
        cursor = conn.cursor()

        try:
            cursor.execute(
                'INSERT INTO users (name, email, number, password, location) VALUES (?, ?, ?, ?, ?)',
                (name, email, number, password, location)
            )
            conn.commit()
            flash('Registration successful!', 'success')
            return redirect(url_for('login'))

        except sqlite3.IntegrityError:
            flash('Email already exists.', 'danger')

        finally:
            conn.close()

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE email = ?', (email,))
        user = cursor.fetchone()

        if user and check_password_hash(user['password'], password):
            session['email'] = user['email']
            flash('Login successful!', 'success')
            return redirect(url_for('home'))
        else:
            flash('Invalid credentials', 'danger')

        conn.close()

    return render_template('login.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True)