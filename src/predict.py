import pickle
from src.feature_extraction import preprocess_code, extract_custom_features
import numpy as np

def predict_language(code):
    """
    Predict the programming language of a code snippet.
    """
    # Load model, vectorizer, and svd
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open("models/svd.pkl", "rb") as f:
        svd = pickle.load(f)

    # Preprocess and extract features
    processed_code = preprocess_code(code)
    X_tfidf = vectorizer.transform([processed_code])
    X_tfidf_reduced = svd.transform(X_tfidf)
    custom_features = extract_custom_features(processed_code)
    X = np.hstack((X_tfidf_reduced, [custom_features]))

    # Predict
    return model.predict(X)[0]