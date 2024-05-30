from flask import Flask, request, render_template
import joblib
from google_play_scraper import reviews_all
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import re

app = Flask(__name__)

# Muat model dan vectorizer
model = joblib.load('knn_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Inisialisasi stemmer Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

def scrape_comments(app_package):
    # Mengumpulkan semua ulasan
    reviews = reviews_all(app_package, lang='id', country='id')
    komentar_list = [review['content'] for review in reviews]
    return komentar_list

def preprocess_text(text):
    # Hilangkan tanda baca dan angka
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Mengubah ke huruf kecil
    text = text.lower()
    # Stemming
    text = stemmer.stem(text)
    return text

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_scrape', methods=['POST'])
def predict_scrape():
    app_package = 'com.kai.kaiticketing'  # Ganti dengan app package yang relevan
    # Lakukan scraping komentar
    komentar_list = scrape_comments(app_package)

    if komentar_list:
        # Preprocess komentar
        komentar_processed = [preprocess_text(komentar) for komentar in komentar_list]

        # Transformasi komentar menggunakan TF-IDF vectorizer
        komentar_tfidf = vectorizer.transform(komentar_processed)

        # Prediksi menggunakan model KNN
        prediksi = model.predict(komentar_tfidf)

        # Buat DataFrame untuk analisis
        df = pd.DataFrame({'komentar': komentar_list, 'prediksi': prediksi})

        # Temukan permasalahan yang paling dominan
        masalah_terbanyak = df['prediksi'].value_counts().idxmax()
        jumlah_masalah = df['prediksi'].value_counts().max()

        return render_template('home.html', masalah=masalah_terbanyak, jumlah=jumlah_masalah)
    else:
        return render_template('home.html', error='Tidak ada komentar yang ditemukan')

@app.route('/comment')
def comment():
    return render_template('comment.html')

@app.route('/predict_comment', methods=['POST'])
def predict_comment():
    komentar = request.form['komentar']
    if komentar:
        # Preprocess komentar
        komentar_processed = preprocess_text(komentar)
        # Transformasi komentar menggunakan TF-IDF vectorizer
        komentar_tfidf = vectorizer.transform([komentar_processed])
        # Prediksi menggunakan model KNN
        prediksi = model.predict(komentar_tfidf)
        return render_template('comment.html', prediction=prediksi[0], komentar=komentar)
    else:
        return render_template('comment.html', error='Please enter a comment')

if __name__ == '__main__':
    app.run(debug=True)
