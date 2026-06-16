# 🤖 Multilingual AI Chatbot with Speech-to-Text and Real-Time Translation

An AI-powered chatbot that enables multilingual conversations, voice interaction, and real-time translation, designed to provide intelligent and accessible communication for users across different languages.

---

## 🚀 Overview

This project is a Flask-based web application integrated with a Deep Learning NLP model that can understand user queries and respond intelligently. It supports both text and voice input, making it more interactive and user-friendly. The system is designed to bridge communication gaps by translating user input and chatbot responses in real-time.

---

## ✨ Key Features

- AI Chatbot – Responds to user queries using trained NLP model  
- Speech-to-Text – Users can interact using voice input  
- Multilingual Support – Real-time translation across multiple languages  
- Intent Recognition – Deep learning-based intent classification  
- User Authentication – Login and registration system  
- Fast & Lightweight – Built with Flask and SQLite  

---

## 🛠️ Tech Stack

Programming: Python  
Backend: Flask  
AI/ML: TensorFlow / Keras, Natural Language Processing (NLP)  
Libraries: NLTK, NumPy, SpeechRecognition, Deep Translator  
Database: SQLite3  
Concepts: Bag of Words, Lemmatization, Intent Classification, Pickle, JSON  

---

## 📂 Project Structure

chat_bot/
│
├── app.py (Main Flask application)  
├── train_chatbot.py (Model training script)  
├── chatbot_model.h5 (Trained deep learning model)  
├── words.pkl (Vocabulary)  
├── classes.pkl (Intent classes)  
├── new.json (Intent dataset)  
├── chatbot.db (SQLite database)  
│
├── templates/  
│   ├── home.html  
│   ├── chatbot.html  
│   ├── login.html  
│   ├── register.html  
│   └── contact.html  
│
├── static/  
│   └── images/  
│
├── key/ (ignored - contains sensitive files)  
└── .gitignore  

---

## ⚙️ Installation & Setup

1. Clone Repository  
git clone https://github.com/chandru-python/Multilingual-AI-Chatbot-with-Speech-to-Text-and-Real-Time-Translation.git  
cd Multilingual-AI-Chatbot-with-Speech-to-Text-and-Real-Time-Translation  

2. Create Virtual Environment  
python -m venv venv  
venv\Scripts\activate  

3. Install Dependencies  
pip install -r requirements.txt  

4. Run Application  
python app.py  

5. Open in Browser  
http://127.0.0.1:5000/  

---

## 🧠 Working Flow

User enters text or voice input  
Speech is converted into text (if voice is used)  
Text is processed using NLP techniques (tokenization, lemmatization, bag of words)  
Model predicts the intent using trained neural network  
Appropriate response is selected  
Response is translated into selected language  
Final output is displayed to the user  

---

## 🔐 Security

Sensitive files such as API keys, JSON credentials, and secret configurations are excluded using .gitignore to ensure security.

---

## 📌 Future Enhancements

Integration with LLMs (ChatGPT) for advanced responses  
Chat history storage  
Mobile responsive UI  
Cloud deployment (AWS / Render / Railway)  
Healthcare-specific intelligent suggestions  

---

## 👨‍💻 Author

Chandru  
AI/ML Developer passionate about building intelligent real-world applications  

---

## ⭐ Support

If you like this project, give it a star on GitHub ⭐
