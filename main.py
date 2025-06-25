import os
from src.train import train
from src.predict import predict_language

def main():
    """
    Main function to handle model training and prediction.
    """
    # Path to data directory
    data_dir = "data/sample_codes"

    print("Loading dataset from:", data_dir)

    # Train model if not already trained
    if not os.path.exists("models/model.pkl"):
        print("Model not found, starting training...")
        train(data_dir)
    else:
        # print("Model already exists, do you want to retrain? (yes/no)")
        response = input("Model already exists, do you want to retrain? (yes/no): ").strip().lower()
        if response == 'yes' or response == 'y':
            train(data_dir)
        else:
            print("Skipping training.")

    # Predict programming language for a new file
    print("Predicting programming language for file...")
    test_file = "test/test.txt"
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8', errors='ignore') as f:
            code = f.read()
        predicted_language = predict_language(code)
        print(f"Predicted programming language: {predicted_language}")
    else:
        print("File does not exist!")

if __name__ == "__main__":
    main()