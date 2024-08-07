import pandas as pd
import sqlite3
import concurrent.futures
import time
import re

from nltk.tokenize import word_tokenize
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.stem import PorterStemmer
# Initialize Sastrawi stopword remover
stopword_factory = StopWordRemoverFactory()
stopword_remover = stopword_factory.create_stop_word_remover()


# Initialize Sastrawi stemmer
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()

# Porter
porter_stemmer = PorterStemmer()

def preprocess_text(text):
    start_time = time.time()
    if not isinstance(text, str):
        return ''
    # clean data
    content_cleaned = re.sub(r'ADVERTISEMENT', '', text)
    # Menghapus data karalter
    content_cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', content_cleaned)
    # Case folding
    content_cleaned  = content_cleaned.lower()
    # Stopwords removal
    content_without_stopwords = stopword_remover.remove(content_cleaned)
    
    # Stemming
    # stemmed_text = stemmer.stem(content_without_stopwords)
    
    # Tokenization
    tokens = word_tokenize(content_without_stopwords)
    
    # # Stemming
    stemmed_words = [porter_stemmer.stem(token) for token in tokens]
    
    # Join the processed tokens back into text
    processed_text = ' '.join(stemmed_words)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds for preprocess_text")
    return processed_text

def insert_into_sqlite(url, title, sentimen):
    conn = sqlite3.connect('prepro.db')
    cursor = conn.cursor()
    try:
        cursor.execute('''INSERT INTO pre_title_baru (url_berita,title,sentimen)
                          VALUES (?, ?, ?)''', (url, title, sentimen))
        conn.commit()
    except sqlite3.IntegrityError:
        print(f"URL '{url}' sudah ada dalam database.")
    finally:
        conn.close()

def get_processed_urls():
    conn = sqlite3.connect('prepro.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT url_berita FROM pre_title_baru''')
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
    content = row['title']
    processed_text = preprocess_text(content)
    
    # Simpan hasil ke dalam SQLite
    insert_into_sqlite(url, processed_text,row['sentimen'])
    print(f"Processed: {url}")
    
    # Meningkatkan jumlah URL yang telah diproses
    url_count.append(url)

def preprocess_texts_parallel(data):
    start_time = time.time()
    processed_urls = get_processed_urls()
    url_count = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            executor.map(preprocess_and_save_text, data, [processed_urls]*len(data), [url_count]*len(data))
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down gracefully...")
        executor.shutdown(wait=False)
    end_time = time.time()
    print(f"Total waktu yang dibutuhkan: {end_time - start_time} detik")
    print(f"Total URL yang telah diproses: {len(url_count)}")

if __name__ == "__main__":
    dataset = pd.read_csv("raw_title.csv")
    preprocess_texts_parallel(dataset.to_dict('records'))