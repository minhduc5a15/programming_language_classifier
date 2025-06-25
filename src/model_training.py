from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import numpy as np
import os

def train_model(X, y, vectorizer, svd):
    """
    Train a classification model using RandomForest.
    Save model, vectorizer, and svd, return trained model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight='balanced'
    )
    model.fit(X, y)

    # Evaluate model using cross-validation
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation accuracy: {np.mean(scores):.4f} (+/- {np.std(scores) * 2:.4f})")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)

    # Save model
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save vectorizer
    with open("models/vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save SVD
    with open("models/svd.pkl", "wb") as f:
        pickle.dump(svd, f)

    return model