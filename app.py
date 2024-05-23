from flask import Flask, request, jsonify, render_template
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Load intents.json to access the responses
with open('intents_IAI.json') as file:
    intents = json.load(file)

def preprocess_text(text):
    # Basic preprocessing: lowercasing and removing non-alphanumeric characters
    return re.sub(r'\W+', ' ', text.lower())

def extract_keywords(texts):
    # Use TF-IDF to identify significant words in queries compared to a corpus
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix

def find_best_response(message, intents, similarity_threshold=0.1):
    preprocessed_message = preprocess_text(message)

    # Evaluate both tags and patterns
    all_texts = [preprocessed_message] + [preprocess_text(item) for intent in intents['intents'] for item in intent['patterns'] + ([intent['tag']] if isinstance(intent['tag'], str) else intent['tag'])]
    tfidf_vectorizer, tfidf_matrix = extract_keywords(all_texts)

    # Calculate similarity scores
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    # Identify the best matching tag or pattern
    max_score = cosine_scores.max()

    if max_score < similarity_threshold:
        return "I don't have any information about that. Please call support for further assistance."

    max_score_index = cosine_scores.argmax()

    # Iterate through intents to find the best match
    response = "I'm not sure how to respond to that."
    current_index = 0
    for intent in intents['intents']:
        combined = intent['patterns'] + ([intent['tag']] if isinstance(intent['tag'], str) else intent['tag'])
        for _ in combined:
            if current_index == max_score_index:
                response = random.choice(intent['responses'])
                return response
            current_index += 1

    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route("/query", methods=["POST"])
def respond_to_query():
    data = request.json
    message = data["message"]
    response = find_best_response(message, intents)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
