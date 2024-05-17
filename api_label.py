import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import requests

API_URL = "https://api-inference.huggingface.co/models/ayameRushia/bert-base-indonesian-1.5G-sentiment-analysis-smsa"
headers = {"Authorization": "masukan token"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def label_sentiment(text):
    try:
        response = query({"inputs": text})
        results = response[0]  # Ambil hasil analisis sentimen dari output
        top_sentiment = max(results, key=lambda x: x['score'])  # Ambil label dengan nilai score tertinggi
        label = top_sentiment['label']
        score = top_sentiment['score']
        print(label, len(text.split()))
        return label, score
    except Exception as e:
        print(f"Error occurred for text: {text}")
        print(f"Error message: {str(e)}")
        return None, None

def label_sentiment_multithread(text_list, max_workers=5):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(label_sentiment, text_list)
    return list(results)

def main():
    data = pd.read_csv('processed_data_all.csv')

    batch_size = 1000
    num_batches = len(data) // batch_size + 1
    batches = [data[i*batch_size:(i+1)*batch_size] for i in range(num_batches)]

    labeled_data = []
    for batch in batches:
        texts = batch['content'].tolist()
        labels = label_sentiment_multithread(texts, max_workers=10)
        labeled_data.extend(labels)

    data['sentiment_label'], data['sentiment_score'] = zip(*labeled_data)

    data.to_csv('label_all.csv', index=False)

if __name__ == "__main__":
    main()
