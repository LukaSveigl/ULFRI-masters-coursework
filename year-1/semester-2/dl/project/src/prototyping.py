from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == '__main__':
    patient_text = "I feel itching and see a skin rash."

    model = KeyBERT('all-MiniLM-L6-v2')

    vectorizer = CountVectorizer(ngram_range=(1, 2))

    vectorizer.fit_transform([patient_text])

    keywords = model.extract_keywords(patient_text, vectorizer=vectorizer, use_mmr=True)

    print(keywords)