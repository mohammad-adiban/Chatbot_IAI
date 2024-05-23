from flask import Flask, request, jsonify, render_template
import json
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import pipeline

app = Flask(__name__)

# Load intents.json to access the responses
with open('intents_IAI.json') as file:
    intents = json.load(file)

# Load a summarization pipeline, adjust model as needed
#summarizer = pipeline("summarization")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def preprocess_text(text):
    # Lowercase and remove non-alphanumeric characters
    return re.sub(r'\W+', ' ', text.lower())

def extract_keywords(texts):
    # Identify significant words in queries
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(texts)
    return tfidf_vectorizer, tfidf_matrix

def find_best_response(message, intents, similarity_threshold=0.1):
    preprocessed_message = preprocess_text(message)
    all_texts = [preprocessed_message] + [preprocess_text(item) for intent in intents['intents'] for item in intent['patterns'] + ([intent['tag']] if isinstance(intent['tag'], str) else intent['tag'])]
    tfidf_vectorizer, tfidf_matrix = extract_keywords(all_texts)
    cosine_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    max_score = cosine_scores.max()

    if max_score < similarity_threshold:
        return "I don't have any information about that. Please call support for further assistance."

    max_score_index = cosine_scores.argmax()
    response = "I'm not sure how to respond to that."
    current_index = 0
    for intent in intents['intents']:
        combined = intent['patterns'] + ([intent['tag']] if isinstance(intent['tag'], str) else intent['tag'])
        for _ in combined:
            if current_index == max_score_index:
                return random.choice(intent['responses'])
            current_index += 1



def preserve_special_content(text):
    replacement_dict = {}
    # Updated pattern keys to ensure uniqueness in the placeholder
    patterns = {
        r'(<code>.*?</code>)': "CODEBLOCK_",
        r'(<pre><code>.*?</pre></code>)': "PRECODEBLOCK_",
        r'(<img>.*?</img>)': "IMAGEBLOCK_"
    }

    for pattern, placeholder_base in patterns.items():
        matches = re.findall(pattern, text, flags=re.DOTALL)
        for i, match in enumerate(matches, start=1):
            # Generate a unique placeholder for each match
            unique_placeholder = f"{placeholder_base}{i}"
            text = text.replace(match, unique_placeholder, 1)  # Replace only the first occurrence to avoid duplicating replacements
            replacement_dict[unique_placeholder] = match
    
    return text, replacement_dict


def restore_special_content(text, replacement_dict):
    for placeholder, original in replacement_dict.items():
        text = text.replace(placeholder, original)
    return text


def clean_text(text):
    """
    Cleans up text by removing unwanted spaces around punctuation, excluding newline patterns.
    """
    # Fix spaces before dots, excluding newline before dot
    text = re.sub(r'(?<!\n)\s+\.', '.', text)
    # Similar for other punctuation marks
    text = re.sub(r'(?<!\n)\s+,', ',', text)
    text = re.sub(r'(?<!\n)\s+!', '!', text)
    text = re.sub(r'(?<!\n)\s+\?', '?', text)
    return text


def summarize_response_based_on_question(question, response):
    preserved_response, replacement_dict = preserve_special_content(response)

    if len(preserved_response.split()) <= 100:
        # Restore and clean for concise responses to avoid over-summarization
        return clean_text(restore_special_content(preserved_response, replacement_dict))
    
    try:
        # Generate a more focused summary for longer responses
        summary = summarizer(f"Question: {question} \n\nAnswer: {preserved_response}", max_length=750, min_length=50, do_sample=True)
        cleaned_summary = clean_text(summary[0]['summary_text'])
        # Restore special content after summarization and cleaning
        return restore_special_content(cleaned_summary, replacement_dict)
    except Exception as e:
        print(f"Error during summarization: {e}")
        # Clean and restore original response if summarization fails
        return clean_text(restore_special_content(preserved_response, replacement_dict))



@app.route('/')
def home():
    return render_template('index.html')

@app.route("/query", methods=["POST"])
def respond_to_query():
    data = request.json
    question = data["message"]
    initial_response = find_best_response(question, intents)
    summarized_response = summarize_response_based_on_question(question, initial_response)
    return jsonify({"response": summarized_response})

if __name__ == "__main__":
    app.run(debug=True)
