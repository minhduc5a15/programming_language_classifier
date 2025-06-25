from src.data_preprocessing import load_data
from src.feature_extraction import extract_features
from src.model_training import train_model
import os

def train(data_dir):
    """
    Train the model using data from the specified directory.
    """
    print("Training model...")
    X, y, vectorizer, svd = load_and_prepare_data(data_dir)
    train_model(X, y, vectorizer, svd)
    print("Training completed!")

def load_and_prepare_data(data_dir):
    """
    Load data and extract features for training.
    """
    codes, labels = load_data(data_dir)
    X, vectorizer, svd = extract_features(codes)
    return X, labels, vectorizer, svd