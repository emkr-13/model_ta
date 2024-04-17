import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import nltk
from nltk.summarization import IndicativeTransformer

def summarize_text(text):
  # Make sure to download the NLTK 'punkt' package
  nltk.download('punkt')

  # Tokenize the text into sentences
  sentences = nltk.sent_tokenize(text)

  # Initialize IndicativeTransformer
  summarizer = IndicativeTransformer()

  # Summarize the text using NLTK
  summary = summarizer.summarize(sentences)

  # Join summary sentences into a string
  summary_text = ' '.join(summary)

  # Return the summary text
  return summary_text

def process_data_chunk(chunk):
    chunk['summary'] = chunk['processed_content'].apply(summarize_text)
    return chunk

def main():
    data = pd.read_csv('lda_content_10000.csv')

    # Dividing the data into several chunks
    chunk_size = 1000
    chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]

    # Using ThreadPoolExecutor to process chunks in parallel
    with ThreadPoolExecutor() as executor:
        processed_chunks = list(executor.map(process_data_chunk, chunks))

    # Combining the processed chunks back together
    processed_data = pd.concat(processed_chunks)

    # Saving the result to a new CSV file
    processed_data.to_csv('summarized_data.csv', index=False)

if __name__ == "__main__":
    main()
