import os
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import warnings

warnings.filterwarnings('ignore')

def main():
    print(">>> Starting Training Module...")
    
    # Mock data directly derived from common job portal texts
    data = {
        'Job_Description': [
            'Software Engineer needed for backend services in Python and Flask.',
            'Earn $10,000 an hour from home! Just send us $20 to register and start earning immediately!',
            'Looking for an enthusiastic Customer Service representative. Full benefits and competitive salary.',
            'URGENT! Provide your bank and social security numbers immediately to apply for this entry-level manager role paying $200k.',
            'Data Analyst required. Must know pandas, React, and Python. Min 2 years exp.',
            'Work closely with top figures. Send 100 dollars to the provided bitcoin wallet. Get rich.',
            'Frontend web developer needed. HTML, CSS, JavaScript, React. Remote work available.',
            'Guaranteed placement in Big Tech Company. Just share your passport copy and credit card info.',
            'We are hiring a systems administrator to manage Linux servers.',
            'Easy money guaranteed. $500 per day cash. No experience, enter your SSN on this unsecured site.'
        ],
        'Is_Fake': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    
    df = pd.DataFrame(data)
    X = df['Job_Description']
    y = df['Is_Fake']
    
    # Define models
    models = {
        'Logistic_Regression': LogisticRegression(),
        'Random_Forest': RandomForestClassifier(n_estimators=10),
        'SVM': SVC(probability=True),
        'Naive_Bayes': MultinomialNB(),
        'Decision_Tree': DecisionTreeClassifier(),
        'KNN': KNeighborsClassifier(n_neighbors=3)
    }
    
    # Ensure models dir exists
    model_dir = os.path.join(os.path.dirname(__file__), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    print("--- Training Pipeline Start ---")
    
    # Train each model using a Pipeline
    for name, model_algo in models.items():
        print(f"Training {name}...")
        pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(stop_words='english', max_features=5000)),
            ('clf', model_algo)
        ])
        
        # Train on the tiny dataset
        pipeline.fit(X, y)
        
        # Predict on same data for logging report (overfit logic just for demo completeness)
        y_pred = pipeline.predict(X)
        acc = accuracy_score(y, y_pred)
        print(f"[{name}] Training Accuracy: {acc*100:.2f}%")
        
        # Save model
        joblib.dump(pipeline, os.path.join(model_dir, f'{name}.pkl'))
        
    print("\n>>> All models trained and saved to 'models/' directory.")

if __name__ == '__main__':
    main()

