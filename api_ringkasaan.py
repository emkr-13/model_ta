import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import time
import requests

API_URL = "https://api-inference.huggingface.co/models/cahya/bert2gpt-indonesian-summarization"
headers = {"Authorization": "Bearer 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"}

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
        payload = {"inputs": text}
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()  # Raise an exception for any HTTP error
        output = response.json()
        # print(output)
        summary_text = output[0].get('summary_text')  # Extract 'summary_text' from the response
        print(summary_text)  # Print the summarized text, not the function
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
    file_path = 'raw_data_baru_100.csv'
    output_file_path = 'summarized_data_100.csv'

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Create a ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=3) as executor:
        try:
            start_process = time.time()
            # Use map to apply the summarize_text function to each text in the dataframe
            summarized_texts = list(executor.map(summarize_text, data['content']))
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
