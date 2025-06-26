from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import pickle
import os

def train_model(X, y, vectorizer, svd):
    """
    Train a classification model using RandomForest.
    Save model, vectorizer, and svd, return trained model.
    """

    param_dist = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', None]
    }

    # Initialize and train the RandomForestClassifier
    base_model = RandomForestClassifier(random_state=42, class_weight='balanced')

    # Use RandomizedSearchCV to find the best parameters
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=20,  # Số lần thử ngẫu nhiên
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=2
    )

    # Train the model
    random_search.fit(X, y)
    model = random_search.best_estimator_
    print(f"Best parameters: {random_search.best_params_}")
    print(f"Best cross-validation score: {random_search.best_score_}")

    # Rate the model using cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")

    # Save the model, vectorizer, and svd
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "model.pkl")
    vectorizer_path = os.path.join(model_dir, "vectorizer.pkl")
    svd_path = os.path.join(model_dir, "svd.pkl")

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(svd_path, 'wb') as f:
        pickle.dump(svd, f)

    return model