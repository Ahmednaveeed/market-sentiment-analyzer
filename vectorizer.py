from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def create_vectorizer(text):
    vectorizer = TfidfVectorizer(
        max_features=1000,                     # Limit to 1000 features, Every tweet becomes a vector of length 1000. 
    )

    x = vectorizer.fit_transform(text)          # Learns vocabulary & transforms each tweet to vector

    with open('Tfidf_vectorizer.pkl', 'wb') as f:     # Save the vectorizer to a file
        pickle.dump(vectorizer, f)   
    
    return x, vectorizer   
