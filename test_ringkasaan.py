import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import time

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

def summarize_text(text, summarization_pipe):
    try:
        summarized_text = summarization_pipe(text)[0]['summary_text']
        return summarized_text
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
    file_path = 'processed_data_all.csv'
    output_file_path = 'summarized_data.csv'

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Initialize the summarization pipeline
    summarization_pipe = pipeline("summarization", model="cahya/bert2gpt-indonesian-summarization")

    # Create a ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor() as executor:
        try:
            start_process = time.time()
            # Use map to apply the summarize_text function to each text in the dataframe
            summarized_texts = list(executor.map(lambda text: summarize_text(text, summarization_pipe), data['content']))
            end_process = time.time()
            print("Texts summarized successfully in {:.2f} seconds.".format(end_process - start_process))
        except Exception as e:
            print("Error occurred during summarization process:", e)
            return

    # Add the summarized texts to the dataframe
    data['summarized_text'] = summarized_texts

    # Save the dataframe to a new CSV file
    save_data(data, output_file_path)

if __name__ == "__main__":
    main()
