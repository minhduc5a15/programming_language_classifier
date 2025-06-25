import os

def load_data(data_dir):
    """
    Load data from the directory containing source code files.
    Returns a list of code snippets and corresponding labels.
    """
    codes = []
    labels = []

    # Iterate through subdirectories (each subdirectory represents a language)
    for language in os.listdir(data_dir):
        language_dir = os.path.join(data_dir, language)
        if os.path.isdir(language_dir):
            for filename in os.listdir(language_dir):
                file_path = os.path.join(language_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                        if len(code) > 100:
                            codes.append(code)
                            labels.append(language)
                except Exception as e:
                    print(f"Error reading file {file_path}: {e}")

    return codes, labels