import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import LdaModel, LsiModel
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel

# Load the data
data = pd.read_csv('data_pemilu.csv')

# Preprocess the data (if necessary)

# Tokenize the content_clean column
documents = data['content_clean'].str.split()

# Create a TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=1,
                                 norm='l2',
                                 smooth_idf=True,
                                 use_idf=True)

# Fit and transform the data
tfidf_matrix = tfidf_vectorizer.fit_transform(data['content_clean'])

# Create a dictionary from the documents
dictionary = Dictionary(documents)

# Create a corpus from the TF-IDF matrix
corpus = [dictionary.doc2bow(doc) for doc in documents]

# Define the range of topics and top words
topics_range = range(1, 11)  # Range of number of topics from 1 to 10
top_words_range = [5]  # Top words per topic: 5 and 10

# Initialize variables to store best coherence scores and corresponding parameters
best_coherence_lda = -1
best_coherence_lsa = -1
best_lda_params = None
best_lsa_params = None

# Iterate through different combinations of topics and top words
for num_topics in topics_range:
    for top_words in top_words_range:
        # Build LDA model
        lda_model = LdaModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics)

        # Build LSI model
        lsi_model = LsiModel(corpus=corpus,
                             id2word=dictionary,
                             num_topics=num_topics)

        # Compute coherence scores for LDA
        coherence_model_lda = CoherenceModel(model=lda_model, texts=documents, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()

        # Compute coherence scores for LSA
        coherence_model_lsa = CoherenceModel(model=lsi_model, texts=documents, dictionary=dictionary, coherence='c_v')
        coherence_lsa = coherence_model_lsa.get_coherence()

        # Print coherence scores for each combination
        print("Number of Topics: {}, Top Words per Topic: {}".format(num_topics, top_words))
        print("Coherence Score (LDA):", coherence_lda)
        print("Coherence Score (LSA):", coherence_lsa)
        print()

        # Update best coherence scores and corresponding parameters if better scores found
        if coherence_lda > best_coherence_lda:
            best_coherence_lda = coherence_lda
            best_lda_params = (num_topics, top_words)

        if coherence_lsa > best_coherence_lsa:
            best_coherence_lsa = coherence_lsa
            best_lsa_params = (num_topics, top_words)

# Print the best coherence scores and corresponding parameters for LDA and LSA
print("Best Coherence Score (LDA):", best_coherence_lda)
print("Best Parameters (LDA): Number of Topics - {}, Top Words per Topic - {}".format(*best_lda_params))
print()
print("Best Coherence Score (LSA):", best_coherence_lsa)
print("Best Parameters (LSA): Number of Topics - {}, Top Words per Topic - {}".format(*best_lsa_params))
