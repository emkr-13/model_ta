import pandas as pd
import joblib
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import hstack

# Load the dataset
data = pd.read_csv('Data_Pemilu.csv')

# Load the model data
loaded_model_data = joblib.load('Sentimen/model_sentimen_svm.pkl')

# Extract model, vectorizer, and label encoder from loaded data
loaded_model = loaded_model_data['model']
tfidf_vectorizer = loaded_model_data['tfidf_vectorizer']
count_vectorizer = loaded_model_data['count_vectorizer']
label_encoder = loaded_model_data['label_encoder']

# Function to predict sentiment for a given text
def predict_sentiment(text):
    # Ensure text is passed as a list to the transform methods
    tfidf_features = tfidf_vectorizer.transform([text])
    count_features = count_vectorizer.transform([text])
    
    # Combine the features
    combined_features = hstack([tfidf_features, count_features])
    
    # Predict sentiment using the loaded model
    prediction = loaded_model.predict(combined_features)
    
    # Decode numerical label back to text label using the loaded LabelEncoder
    sentiment = label_encoder.inverse_transform(prediction)[0]
    print(sentiment)
    
    return sentiment

# Function to process each row in the dataframe and predict sentiment
def process_row(row):
    content = row['content']
    sentiment = predict_sentiment(content)
    return sentiment

# Process the dataframe using threads
with ThreadPoolExecutor(max_workers=8) as executor:
    data['sentimen'] = list(executor.map(process_row, [row for _, row in data.iterrows()]))

# Save the updated dataframe to a new CSV file
data.to_csv('data_pemilu_label.csv', index=False)
