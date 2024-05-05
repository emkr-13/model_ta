import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor

# Load the dataset
data = pd.read_csv('online_news_50000_clean_all.csv')

# Load the model data
loaded_model_data = joblib.load('Sentimen/logistic_regression_with_vectorizer_baru.pkl')

# Extract model, vectorizer, and label encoder from loaded data
loaded_model = loaded_model_data['model']
loaded_vectorizer = loaded_model_data['vectorizer']
loaded_label_encoder = loaded_model_data['label_encoder']

# Function to predict sentiment for a given text
def predict_sentiment(text):
    # Vectorize the text using the loaded CountVectorizer
    text_features = loaded_vectorizer.transform([text])
    # Predict sentiment using the loaded model
    prediction = loaded_model.predict(text_features)
    # Decode numerical label back to text label using the loaded LabelEncoder
    sentiment = loaded_label_encoder.inverse_transform(prediction)[0]
    return sentiment

# Function to process each row in the dataframe and predict sentiment
def process_row(row):
    content = row['content']
    sentiment = predict_sentiment(content)
    print(sentiment)
    return sentiment

# Number of threads to use
num_threads = 8  # Adjust as needed

# Split the dataframe into chunks for parallel processing
chunks = [data[i:i+len(data)//num_threads] for i in range(0, len(data), len(data)//num_threads)]

# Process each chunk using threads
with ThreadPoolExecutor(max_workers=num_threads) as executor:
    # Map each chunk to the process_row function using threads
    results = executor.map(lambda chunk: chunk.apply(process_row, axis=1), chunks)

# Concatenate the results from all chunks
predicted_sentiments = pd.concat(results)

# Add the predicted sentiments to the original dataframe
data['sentimen'] = predicted_sentiments

# Save the updated dataframe to a new CSV file
data.to_csv('online_news_50000_clean_label_all_baru.csv', index=False)
