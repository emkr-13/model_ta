import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline

def label_sentiment(text):
    try:
        # Analisis sentimen menggunakan model Transformer
        result = sentiment_analysis(text)
        # Ambil label sentimen teratas
        top_sentiment = result[0]
        label = top_sentiment['label']
        score = top_sentiment['score']
        print(label,score,len(text.split()))
        return label, score
    except Exception as e:
        print(f"Error occurred for text: {text}")
        print(f"Error message: {str(e)}")
        return None, None

def label_sentiment_multithread(text_list, max_workers=5):  # default 5 thread jika tidak disebutkan
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(label_sentiment, text_list)
    return list(results)

def main():
    data = pd.read_csv('data_pemilu_filter.csv')
    # Load model analisis sentimen bahasa Indonesia dengan tiga kelas sentimen
    global sentiment_analysis
    sentiment_analysis = pipeline("text-classification", model="taufiqdp/indonesian-sentiment")
    # Bagi data menjadi batch agar dapat diproses menggunakan multi-threading
    batch_size = 1000
    num_batches = len(data) // batch_size + 1
    batches = [data[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]
    # Melabeli sentimen untuk setiap batch menggunakan multi-threading
    labeled_data = []
    for batch in batches:
        texts = batch['content'].tolist()
        labels = label_sentiment_multithread(texts, max_workers=10)  # Misalnya, menggunakan 10 thread
        labeled_data.extend(labels)
    # Memasukkan hasil label sentimen ke dalam DataFrame
    data['sentimen'], data['sentimen_score'] = zip(*labeled_data)
    # Simpan data yang telah dilabeli ke dalam file CSV
    data.to_csv('data_pemilu_label.csv', index=False)

if __name__ == "__main__":
    main()
