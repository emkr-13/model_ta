import pandas as pd
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
    start_time = time.time()
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
    end_time = time.time()
    print(f"Total waktu yang dibutuhkan: {end_time - start_time} detik untuk preprocess_text")
    return processed_text
    
def preprocess_and_save_text(row, processed_urls):
    url = row['link_berita']
    
    # Mengecek apakah teks sudah diproses sebelumnya berdasarkan URL
    csv_file = 'processed_data.csv'
    if url in processed_urls:
        print(f"Skip: {url} already processed")
        return
    
    # Jika teks belum diproses, lakukan preprocessing
    content = row['content']
    processed_text = preprocess_text(content)
    
    # Simpan hasil ke dalam CSV
    row_dict = {
        'nama_berita': row['nama_berita'],
        'tanggal_berita': row['tanggal_berita'],
        'link_berita': url,
        'processed_text': processed_text
    }
    with open(csv_file, 'a') as f:
        pd.DataFrame([row_dict]).to_csv(f, header=not os.path.isfile(csv_file), index=False)
    print(f"Processed: {url}")
        
def preprocess_texts_parallel(data):
    start_time = time.time()
    processed_urls = set()
    csv_file = 'processed_data.csv'
    if os.path.isfile(csv_file):
        df = pd.read_csv(csv_file)
        processed_urls = set(df['link_berita'].values)  # Memperbaiki pengambilan kolom yang sesuai
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            executor.map(preprocess_and_save_text, data, [processed_urls]*len(data))
    except KeyboardInterrupt:
        print("Keyboard interrupt received. Shutting down gracefully...")
        executor.shutdown(wait=False)
    end_time = time.time()
    print(f"Total waktu yang dibutuhkan: {end_time - start_time} detik")
    
if __name__ == "__main__":
    dataset = pd.read_csv("unique_data_content.csv")
    preprocess_texts_parallel(dataset.to_dict('records'))
