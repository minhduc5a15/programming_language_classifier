from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import numpy as np
from src.language_config import LANGUAGE_CONFIG, STOP_WORDS


def extract_features(codes, n_components=200):
    """
    Extract features from code snippets using TF-IDF and custom features.
    Args:
        codes (list): List of code snippets as strings.
        n_components (int): Number of SVD components for dimensionality reduction.
    Returns:
        tuple: (X, vectorizer, svd)
            - X (np.ndarray): Feature matrix combining TF-IDF and custom features.
            - vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
            - svd (TruncatedSVD): Fitted SVD transformer.
    """
    if not codes:
        raise ValueError("No code snippets provided for feature extraction")

    # Preprocess code snippets
    processed_codes = [preprocess_code(code) for code in codes]

    # TF-IDF Vectorizer
    vocabulary = set()
    for config in LANGUAGE_CONFIG.values():
        vocabulary.update(config["keywords"])
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 4),
        token_pattern=r"\b\w+(?:\.\w+)*\b|::|->|=>|[.;{}()#@-]|[*]{1,2}|&&|\||[<>=!]=|[-+*/%]=|~|\^|\|",
        stop_words=[w for w in STOP_WORDS if w.strip()],
        lowercase=False,  # Preserve case for keywords like True, None
        vocabulary=list(vocabulary)  # Prioritize language keywords
    )
    X_tfidf = vectorizer.fit_transform(processed_codes)

    # Adjust n_components to be <= number of features
    n_features = X_tfidf.shape[1]
    n_components = min(n_components, n_features)
    if n_features < 10:
        print(f"Warning: Only {n_features} TF-IDF features extracted. Consider adding more data or adjusting max_features.")

    # Dimensionality reduction using SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_tfidf_reduced = svd.fit_transform(X_tfidf)

    # Extract custom features
    custom_features = np.array([extract_custom_features(code) for code in processed_codes])

    # Combine TF-IDF features with custom features
    X = np.hstack((X_tfidf_reduced, custom_features))

    # Debugging training information
    print(f"Number of TF-IDF features: {n_features}")
    print(f"Sample TF-IDF features: {list(vectorizer.get_feature_names_out())[:10]}")
    print(f"Number of SVD components: {n_components}")
    print(f"Total features after combining: {X.shape[1]}")

    return X, vectorizer, svd


def preprocess_code(code):
    """
    Preprocess code snippet: remove comments, normalize whitespace, retain key structures.
    Args:
        code (str): Raw code snippet.
    Returns:
        str: Preprocessed code snippet.
    """
    # Keep specific comment patterns
    comment_patterns = [r'@Override\s', r'@[A-Z]\w+\s', r'#pragma\s+\w+']
    comments = ' '.join(m.group() for p in comment_patterns for m in re.finditer(p, code))
    # Remove all comments
    code = re.sub(r"//.*?\n|/\*.*?\*/", "", code, flags=re.DOTALL)
    # Add back relevant comments
    code = comments + ' ' + code if comments else code
    # Remove long non-code strings (e.g., JSON, XML)
    code = re.sub(r'".{50,}"|\'.{50,}\'|[\w+/=]{50,}', "", code)
    # Normalize whitespace
    code = re.sub(r"\s+", " ", code).strip()
    return code if code else " "  # Return single space if empty to avoid TfidfVectorizer error


def extract_custom_features(code):
    """
    Extract custom features: keyword counts, syntax structures, variable name length, patterns.
    Args:
        code (str): Preprocessed code snippet.
    Returns:
        list: Feature vector containing keyword counts, pattern counts, syntax counts,
              average variable name length, and naming conventions.
    """
    # Count language-specific keywords
    keyword_counts = []
    pattern_counts = []
    for lang, config in LANGUAGE_CONFIG.items():
        # Keyword counts
        keyword_count = sum(code.count(kw) for kw in config["keywords"])
        keyword_counts.append(keyword_count)
        # Pattern counts
        pattern_count = sum(len(re.findall(p, code)) for p in config["patterns"])
        pattern_counts.append(pattern_count)

    # Count syntax structures
    syntax_counts = [
        code.count("{"), code.count("}"), code.count(";"),
        code.count("("), code.count(")"), code.count("for"),
        code.count("while"), code.count("if"), code.count("else"),
        code.count("::"), code.count("->"), code.count("=>"),
    ]

    # Calculate average variable name length
    words = re.findall(r"\b\w+\b", code)
    avg_name_length = np.mean([len(word) for word in words]) if words else 0

    # Calculate naming conventions (snake_case, camelCase)
    snake_case = sum(1 for w in words if "_" in w and w.islower())
    camel_case = sum(1 for w in words if any(c.isupper() for c in w[1:]))
    naming_features = [
        snake_case / len(words) if words else 0,
        camel_case / len(words) if words else 0
    ]

    # Combine all features
    return keyword_counts + pattern_counts + syntax_counts + [avg_name_length] + naming_features
