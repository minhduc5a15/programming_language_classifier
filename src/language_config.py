# Configuration for language-specific keywords, patterns, and stop words
# Add new languages by extending LANGUAGE_CONFIG dictionary

LANGUAGE_CONFIG = {
    "Python": {
        "keywords": [
            "def", "class", "import", "from", 'if __name__ == "__main__"',
            "lambda", "None", "True", "False"
        ],
        "patterns": [
            r"\[\w+\s+for\s+\w+\s+in\s+[^]]+\]"  # List comprehension
        ]
    },
    "Java": {
        "keywords": [
            "class", "public", "private", "static", "void", "interface",
            "extends", "implements"
        ],
        "patterns": [
            r"@Override\s", r"@[A-Z]\w+\s"  # Annotations
        ]
    },
    "C": {
        "keywords": [
            "#include <stdio.h>", "#include <stdlib.h>", "printf", "scanf",
            "struct", "typedef", "int main", "void main(", "getchar",
            "putchar", "#define"
        ],
        "patterns": [
            r"#include\s+<[\w\.]+>"  # Include directives
        ]
    },
    "Cpp": {
        "keywords": [
            "class", "public:", "private:", "protected:", "namespace", "std::",
            "cout", "cin", "iostream", "template<typename", "virtual", "override",
            "using namespace std", "vector", "string", "#include <vector>",
            "#include <string>", "#include <iostream>", "#include <algorithm>",
            "#include <map>", "#include <set>", "#include <unordered_map>"
        ],
        "patterns": [
            r"std::\w+", r"template\s*<\s*typename\s+\w+\s*>"  # STL and templates
        ]
    },
    "JavaScript": {
        "keywords": [
            "function", "var", "let", "const", "=>", "console.log",
            "module.exports"
        ],
        "patterns": [
            r"async\s+function", r"await\s+"  # Async/await
        ]
    },
    "TypeScript": {
        "keywords": [
            "function", "var", "let", "const", "=>", "console.log",
            "module.exports", "interface", "type", "readonly", ": number",
            ": string", ": boolean", ": any", ": undefined", ": unknown",
            "Omit<", "Partial<", "Pick<", "Record<", "Awaited<", "Promise<",
            "Readonly<", "Exclude<", "NonNullable<", "ConstructorParameters<",
            "ReturnType<", "InstanceType<"
        ],
        "patterns": [
            r"\b\w+\s*:\s*(number|string|boolean|any|void|unknown|never|undefined|bigint|symbol|object)",
            r"interface\s+\w+\s*{",
            r"type\s+\w+\s*="
        ]
    },
    "Rust": {
        "keywords": [
            "fn", "let", "mut", "struct", "enum", "impl", "match",
            "if let", "while let"
        ],
        "patterns": [
            r"->\s*\w+", r"\w+\s*:\s*\w+"  # Return types and type annotations
        ]
    },
    "Go": {
        "keywords": [
            "func", "package", "import", "var", "const", "type", "struct",
            "interface", "defer", "go"
        ],
        "patterns": [
            r"func\s+\w+\s*\([^)]*\)\s*{", r"go\s+\w+"  # Functions and goroutines
        ]
    }
}

# Common stop words across all languages
STOP_WORDS = ["void", "string", "if", "else", "for", "while"]

# Example of adding a new language:
# LANGUAGE_CONFIG["Ruby"] = {
#     "keywords": ["def", "class", "module", "require", "end", "self"],
#     "patterns": [r"do\s*\|[\w,\s]+\|", r"@\w+"]  # Blocks and instance variables
# }
