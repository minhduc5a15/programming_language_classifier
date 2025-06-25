from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import re
import numpy as np

# Language-specific keywords and structures
# These keywords are used to identify the language and extract features
LANGUAGE_KEYWORDS = {
    "Python": [
        "def",
        "class",
        "import",
        "from",
        'if __name__ == "__main__"',
        "lambda",
        "None",
        "True",
        "False",
    ],
    "Java": [
        "class",
        "public",
        "private",
        "static",
        "void",
        "interface",
        "extends",
        "implements",
    ],
    "C": [
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "printf",
        "scanf",
        "struct",
        "typedef",
        "int main(",
        "void main(",
        "getchar",
        "putchar",
        "#define",
    ],
    "Cpp": [
        "class",
        "public:",
        "private:",
        "protected:",
        "namespace",
        "std::",
        "cout",
        "cin",
        "iostream",
        "template<typename",
        "virtual",
        "override",
        "using namespace std",
        "vector",
        "string",
        "#include <vector>",
        "#include <string>",
        "#include <iostream>",
        "#include <algorithm>",
        "#include <map>",
        "#include <set>",
        "#include <unordered_map>",
    ],
    "JavaScript": [
        "function",
        "var",
        "let",
        "const",
        "=>",
        "console.log",
        "module.exports",
    ],
    "TypeScript": [
        "function",
        "var",
        "let",
        "const",
        "=>",
        "console.log",
        "module.exports",
        "interface",
        "type",
        "readonly",
        ": number",
        ": string",
        ": boolean",
        ": any",
        ": undefined",
        ": unknown",
        "Omit<",
        "Partial<",
        "Pick<",
        "Record<",
        "Awaited<",
        "Promise<",
        "Readonly<",
        "Exclude<",
        "NonNullable<",
        "ConstructorParameters<",
        "ReturnType<",
        "InstanceType<",
    ],
    "Rust": [
        "fn",
        "let",
        "mut",
        "struct",
        "enum",
        "impl",
        "match",
        "if let",
        "while let",
    ],
    "Go": [
        "func",
        "package",
        "import",
        "var",
        "const",
        "type",
        "struct",
        "interface",
        "defer",
        "go",
    ],
}

# Common stop words to ignore in code snippets
STOP_WORDS = ["int", "void", "main", "string", "return", "if", "else", "for", "while"]


def extract_features(codes, n_components=100):
    """
    Extract features from code snippets using TF-IDF and custom features.
    Returns feature matrix, vectorizer, and SVD transformer.
    """
    # Preprocess code snippets
    processed_codes = [preprocess_code(code) for code in codes]

    # TF-IDF Vectorizer
    vectorizer = TfidfVectorizer(
        max_features=25000,  # Increase max features
        ngram_range=(1, 5),  # Expand n-grams to include up to 5-grams
        token_pattern=r"\b\w+(?:\.\w+)*\b|::|->|[.;{}():#@-]|[*]{1,2}|&&|\|\|",  # Thêm toán tử
        stop_words=STOP_WORDS,
    )
    X_tfidf = vectorizer.fit_transform(processed_codes)

    # Dimensionality reduction using SVD
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    X_tfidf_reduced = svd.fit_transform(X_tfidf)

    # Extract custom features
    custom_features = np.array(
        [extract_custom_features(code) for code in processed_codes]
    )

    # Combine TF-IDF features with custom features
    X = np.hstack((X_tfidf_reduced, custom_features))

    # Debug training information
    print(f"Number of TF-IDF features: {len(vectorizer.get_feature_names_out())}")
    print(f"Sample TF-IDF features: {list(vectorizer.get_feature_names_out())[:20]}")
    print(f"Number of SVD components: {n_components}")
    print(f"Total features after combining: {X.shape[1]}")

    return X, vectorizer, svd


def preprocess_code(code):
    """
    Preprocess code snippet: remove comments, normalize whitespace, retain key structures.
    """
    # Remove C/C++ style comments (// and /* */)
    code = re.sub(r"//.*?\n|/\*.*?\*/", "", code, flags=re.DOTALL)
    # Remove Python style comments (#)
    code = re.sub(r"#.*?\n", "", code)
    # Remove long non-code strings (e.g., JSON, XML, base64)
    code = re.sub(r'".{50,}"|\'.{50,}\'|[\w+/=]{50,}', "", code)
    # Normalize whitespace while preserving important symbols
    code = re.sub(r"\s+", " ", code).strip()
    return code


def extract_custom_features(code):
    """
    Extract custom features: keyword counts, syntax structures, variable name length.
    Returns a feature vector.
    """
    # Count language-specific keywords
    # This counts the occurrences of each keyword in the code snippet
    keyword_counts = []
    for lang, keywords in LANGUAGE_KEYWORDS.items():
        count = sum(code.count(kw) for kw in keywords)
        keyword_counts.append(count)

    # Count syntax structures
    syntax_counts = [
        code.count("{"),
        code.count("}"),
        code.count(";"),
        code.count("("),
        code.count(")"),
        code.count("for"),
        code.count("while"),
        code.count("if"),
        code.count("else"),
    ]

    # Calculating average variable name length
    words = re.findall(r"\b\w+\b", code)
    avg_name_length = np.mean([len(word) for word in words]) if words else 0

    # Combine features
    return keyword_counts + syntax_counts + [avg_name_length]
