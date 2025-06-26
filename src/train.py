from sklearn.metrics import confusion_matrix

from src.data_preprocessing import load_data
from src.feature_extraction import extract_features
from src.model_training import train_model
import matplotlib.pyplot as plt
import seaborn as sns

def train(data_dir):
    """
    Train the model using data from the specified directory.
    """
    print("Training model...")
    X, y, vectorizer, svd = load_and_prepare_data(data_dir)
    model = train_model(X, y, vectorizer, svd)
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    plt.figure(figsize=(10, 8))

    sns.heatmap(cm, annot=True, fmt="d", xticklabels=model.classes_, yticklabels=model.classes_)

    plt.xlabel('Predicted')
    plt.ylabel('True')

    plt.title('Confusion Matrix')

    plt.savefig('confusion_matrix.png')
    plt.show()

    print("Training completed!")

def load_and_prepare_data(data_dir):
    """
    Load data and extract features for training.
    """
    codes, labels = load_data(data_dir)
    X, vectorizer, svd = extract_features(codes)
    return X, labels, vectorizer, svd