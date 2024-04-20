import pandas as pd
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
    content_cleaned=content_cleaned.replace("JAKARTA, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("DEPOK, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BADUNG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("DEPOK, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BALI, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BALIKPAPAN, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BANDUNG BARAT, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BANDUNG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BANTEN, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BANYUWANGI, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BATANG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BEKASI, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("BOGOR KOMPAS.com", " ")
    content_cleaned=content_cleaned.replace("BOYOLALI, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("CILEGON, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("CIREBON, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("DEPOK, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("GRESIK, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MADIUN, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MAGELANG KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MAGETAN, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MALANG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MAKASSAR, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MEDAN, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("MERAUKE, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("PADANG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("PALU, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("PURBALINGGA, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("PURWOKERTO, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("PURWOREJO, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("SURABAYA, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("SURAKARTA, KOMPAS.com", " ")
    content_cleaned=content_cleaned.replace("TANGERANG SELATAN, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("TANGERANG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("TASIKMALAYA, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("TANGERANG, KOMPAS.com", "")
    content_cleaned=content_cleaned.replace("selengkapnya", "")
    content_cleaned = re.sub(r'Detik News', '', content_cleaned)
    content_cleaned = re.sub(r'CNN News', '', content_cleaned)
    content_cleaned = re.sub(r'KOMPAS.com', '', content_cleaned)
    content_cleaned = re.sub(r'Kompas News', '', content_cleaned)
    content_cleaned = re.sub(r'Gambas', '', content_cleaned)
    content_cleaned = re.sub(r'20detik', '', content_cleaned)
    content_cleaned = re.sub(r'berikutnya', '', content_cleaned)
    content_cleaned = re.sub(r'halaman', '', content_cleaned)
    content_cleaned = re.sub(r'detikcom', '', content_cleaned)
    content_cleaned = re.sub(r'Halaman', '', content_cleaned)
    # Menghapus data karalter
    content_cleaned = re.sub(r'[^a-zA-Z0-9\s]+', ' ', content_cleaned)
    # Case folding
    content_cleaned  = content_cleaned.lower()
    # Stopwords removal
    content_without_stopwords = stopword_remover.remove(content_cleaned)
    
    # Stemming
    # stemmed_text = stemmer.stem(content_without_stopwords)
    
    # Tokenization
    # tokens = word_tokenize(content_without_stopwords)
    
    # # Stemming
    # stemmed_words = [porter_stemmer.stem(token) for token in tokens]
    
    # Join the processed tokens back into text
    # processed_text = ' '.join(tokens)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds for preprocess_text")
    return content_without_stopwords

def preprocess_and_save_text(text):
    processed_text = preprocess_text(text)
    return processed_text

def preprocess_texts_parallel(data):
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(preprocess_and_save_text, data))
    df = pd.DataFrame({'processed_content': results})
    df.to_csv('data_baru_all.csv', index=False)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    dataset = pd.read_csv('raw_data_baru_content.csv')
    preprocess_texts_parallel(dataset['content'])