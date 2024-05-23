from flask import Flask, request, jsonify, render_template
import pickle
import re
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

app = Flask(__name__)

# Muat model KNN dan vectorizer dari file
with open('knn_model_final.pkl', 'rb') as model_file:
    knn_model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vec_file:
    vectorizer = pickle.load(vec_file)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text

def stem_text(text):
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    return stemmer.stem(text)

# Mengambil daftar stopwords
stopword_factory = StopWordRemoverFactory()
stopwords = stopword_factory.get_stop_words()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comment = data['comment']
    cleaned_comment = clean_text(comment)
    stemmed_comment = stem_text(cleaned_comment)
    words = stemmed_comment.split()
    filtered_words = [word for word in words if word not in stopwords]
    filtered_comment = ' '.join(filtered_words)
    vectorized_comment = vectorizer.transform([filtered_comment])
    prediction = knn_model.predict(vectorized_comment)
    return jsonify({'prediction': int(prediction[0])})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
