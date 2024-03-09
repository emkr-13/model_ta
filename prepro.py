import pandas as pd
import sqlite3
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import os.path
import concurrent.futures
import time

# Inisialisasi stemmer dari Sastrawi
factory = StemmerFactory()
stemmer = factory.create_stemmer()

nltk.download('punkt')
nltk.download('stopwords')

# Mengunduh stopwords bahasa Indonesia
stop_words = set(stopwords.words('indonesian'))

def preprocess_text(text):
    if not isinstance(text, str):
        return ''
    # Tokenisasi
    tokens = word_tokenize(text)
    
    # Penghapusan stopwords dan stemming
    cleaned_tokens = []
    for token in tokens:
        # Hanya mempertimbangkan token yang bukan stopwords
        if token.lower() not in stop_words:
            # Melakukan stemming pada token
            stemmed_token = stemmer.stem(token)
            cleaned_tokens.append(stemmed_token)
    
    # Gabungkan kembali token yang telah diproses menjadi teks
    processed_text = ' '.join(cleaned_tokens)
    return processed_text

def insert_into_sqlite(url, nama_berita, tanggal_berita, processed_text):
    conn = sqlite3.connect('prepro.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO pre_content (url_berita, nama_berita, tanggal_berita, processed_text)
                          VALUES (?, ?, ?, ?)''', (url, nama_berita, tanggal_berita, processed_text))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"URL '{url}' sudah ada dalam database.")
    finally:
        conn.close()

def get_processed_urls():
    conn = sqlite3.connect('prepro.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT url_berita FROM pre_content''')
    processed_urls = set(row[0] for row in cursor.fetchall())
    conn.close()
    return processed_urls

def preprocess_and_save_text(row, processed_urls, url_count):
    url = row['link_berita']
    
    # Mengecek apakah teks sudah diproses sebelumnya berdasarkan URL
    if url in processed_urls:
        print(f"Skip: {url} already processed")
        return
    
    # Jika teks belum diproses, lakukan preprocessing
    content = row['content']
    processed_text = preprocess_text(content)
    
    # Simpan hasil ke dalam SQLite
    insert_into_sqlite(url, row['nama_berita'], row['tanggal_berita'], processed_text)
    print(f"Processed: {url}")
    
    # Meningkatkan jumlah URL yang telah diproses
    url_count.append(url)

def preprocess_texts_parallel(data):
    start_time = time.time()
    processed_urls = get_processed_urls()
    url_count = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(preprocess_and_save_text, data, [processed_urls]*len(data), [url_count]*len(data))
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down gracefully...")
        executor.shutdown(wait=False)
    end_time = time.time()
    print(f"Total waktu yang dibutuhkan: {end_time - start_time} detik")
    print(f"Total URL yang telah diproses: {len(url_count)}")

if __name__ == "__main__":
    dataset = pd.read_csv("unique_data_content.csv")
    preprocess_texts_parallel(dataset.to_dict('records'))
