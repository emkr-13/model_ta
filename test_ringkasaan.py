import pandas as pd
from transformers import pipeline
from concurrent.futures import ThreadPoolExecutor
import time
from transformers import BertTokenizer, EncoderDecoderModel

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
        start_load = time.time()
        tokenizer = BertTokenizer.from_pretrained("cahya/bert2gpt-indonesian-summarization")
        tokenizer.bos_token = tokenizer.cls_token
        tokenizer.eos_token = tokenizer.sep_token
        model = EncoderDecoderModel.from_pretrained("cahya/bert2gpt-indonesian-summarization")


        # generate summary
        input_ids = tokenizer.encode(text, return_tensors='pt')
        summary_ids = model.generate(input_ids,
                        min_length=20,
                        max_length=80, 
                        num_beams=10,
                        repetition_penalty=2.5, 
                        length_penalty=1.0, 
                        early_stopping=True,
                        no_repeat_ngram_size=2,
                        use_cache=True,
                        do_sample = True,
                        temperature = 0.8,
                        top_k = 50,
                        top_p = 0.95)

        summarize_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        end_load = time.time()
        print("Text summirized successfully in {:.2f} seconds.".format(end_load - start_load))
        return summarize_text
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
    output_file_path = 'summarized_data.csv'

    # Load the data
    data = load_data(file_path)
    if data is None:
        return

    # Initialize the summarization pipeline
    summarization_pipe = pipeline("summarization", model="cahya/bert2gpt-indonesian-summarization")

    # Create a ThreadPoolExecutor for multithreading
    with ThreadPoolExecutor(max_workers=3) as executor:
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
    end_load = time.time()
    print("Waktu yang di butuh selama  successfully in {:.2f} seconds.".format(end_load - start_load))

if __name__ == "__main__":
    main()
