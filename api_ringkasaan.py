import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import requests

API_URL = "https://api-inference.huggingface.co/models/cahya/bert2gpt-indonesian-summarization"
headers = {"Authorization": "masukan token"}

# Define the maximum requests per minute and the time interval to spread the requests
MAX_REQUESTS_PER_MINUTE = 1000
REQUEST_SPREAD_INTERVAL = 60 / MAX_REQUESTS_PER_MINUTE

def load_data(file_path):
    try:
        start_load = time.time()
        data = pd.read_csv(file_path)
        end_load = time.time()
        print("Data loaded successfully in {:.2f} seconds.".format(end_load - start_load))
        return data
    except Exception as e:
        print("Error occurred while loading the data:", e)
        return None

def summarize_text(text):
    try:
        start_save = time.time()
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for any HTTP error
        output = response.json()
        summary_text = output[0].get('summary_text')  # Extract 'summary_text' from the response
        end_save = time.time()
        print("time use for in {:.2f} seconds.".format(end_save - start_save))
        return summary_text
    except Exception as e:
        print("Error occurred while summarizing text:", e)
        return None


def save_data(data, file_path):
    try:
        start_save = time.time()
        data.to_csv(file_path, index=False)
        end_save = time.time()
        print("Summarized data saved successfully in {:.2f} seconds.".format(end_save - start_save))
    except Exception as e:
        print("Error occurred while saving the summarized data:", e)

def main():
    start_load = time.time()
    file_path = 'processed_data_baru_300.csv'
    output_file_path = 'summarized_data_30000.csv'

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Create a ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=8) as executor:
        try:
            start_process = time.time()
            summarized_texts = []
            
            # Control the rate of requests
            requests_count = 0
            for text in data['content']:
                if requests_count > 0 and requests_count % MAX_REQUESTS_PER_MINUTE == 0:
                    time.sleep(REQUEST_SPREAD_INTERVAL)  # Spread the requests
                summary = summarize_text(text)
                if summary is not None:
                    summarized_texts.append(summary)
                requests_count += 1
            
            end_process = time.time()
            print("Texts summarized successfully in {:.2f} seconds.".format(end_process - start_process))
        except Exception as e:
            print("Error occurred during summarization process:", e)
            return

    # Add the summarized texts to the dataframe
    data['summarized_text'] = summarized_texts

    # Save the dataframe to a new CSV file
    save_data(data, output_file_path)
    end_load = time.time()
    print("Waktu yang di butuh selama  successfully in {:.2f} seconds.".format(end_load - start_load))

if __name__ == "__main__":
    main()
