import pandas as pd
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool

def calculateSimilarity(sentence, document):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence] + document)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_matrix[0][0]

def summarize_sentences(sentence):
    temp_doc = [s for s in sentences if s != sentence]
    score = calculateSimilarity(sentence, temp_doc)
    print(sentence,score)
    return sentence, score

if __name__ == '__main__':
    data = pd.read_csv('lda_content_10000.csv')
    sentences = data['processed_content'].tolist()

    with Pool() as pool:
        scores = dict(pool.map(summarize_sentences, sentences))

    n = int(20 * len(sentences) / 100)
    alpha = 0.5
    summarySet = []

    while n > 0:
        mmr = {}
        
        for sentence in scores.keys():
            if sentence not in summarySet:
                if summarySet:  # Check if summarySet is not empty
                    mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence, summarySet)
                else:
                    mmr[sentence] = alpha * scores[sentence]
        
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        n -= 1

    summary_sentences = [sentence.lstrip(' ') for sentence in summarySet]

    summary_df = pd.DataFrame({'Summary': summary_sentences})
    summary_df.to_csv('summary.csv', index=False)
    print('Summary saved to summary.csv')
import pandas as pd
import operator
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Pool

def calculateSimilarity(sentence, document):
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform([sentence] + document)
    similarity_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])
    return similarity_matrix[0][0]

def summarize_sentences(sentence):
    temp_doc = [s for s in sentences if s != sentence]
    score = calculateSimilarity(sentence, temp_doc)
    print(sentence)
    return sentence, score

if __name__ == '__main__':
    data = pd.read_csv('raw_data_baru_content_5000.csv')
    sentences = data['content'].tolist()

    with Pool() as pool:
        scores = dict(pool.map(summarize_sentences, sentences))

    n = int(20 * len(sentences) / 100)
    alpha = 0.5
    summarySet = []

    while n > 0:
        mmr = {}
        
        for sentence in scores.keys():
            if sentence not in summarySet:
                if summarySet:  # Check if summarySet is not empty
                    mmr[sentence] = alpha * scores[sentence] - (1 - alpha) * calculateSimilarity(sentence, summarySet)
                else:
                    mmr[sentence] = alpha * scores[sentence]
        
        selected = max(mmr.items(), key=operator.itemgetter(1))[0]
        summarySet.append(selected)
        n -= 1

    summary_sentences = [sentence.lstrip(' ') for sentence in summarySet]

    summary_df = pd.DataFrame({'Summary': summary_sentences})
    summary_df.to_csv('summary.csv', index=False)
    print('Summary saved to summary.csv')
