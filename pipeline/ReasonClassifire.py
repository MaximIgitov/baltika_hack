# import sys
# sys.path.append(r'D:\repositories\baltika_hack')

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import optuna
import pickle


class ProblemClassifier:
    def __init__(self, model_path: str = 'pipeline/models/reason_classifier.pkl'):
        self.model_path = model_path
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = None

    def fit_reason_classifier(self, data_path: str = 'pipeline/resources/Categorical breakdowns (V4).xlsx'):
        data = pd.read_excel(data_path)
        X = data['reason'].tolist()
        y = data['reason_short']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        def objective(trial):
            C = trial.suggest_loguniform('C', 1e-4, 1e2)
            model = LogisticRegression(C=C, multi_class='ovr', solver='lbfgs', max_iter=1000)
            model.fit(X_train_vec, y_train)
            y_pred = model.predict_proba(X_test_vec)
            return roc_auc_score(pd.get_dummies(y_test), y_pred, multi_class='ovr')

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=50)

        best_model = LogisticRegression(C=study.best_params['C'], multi_class='ovr', solver='lbfgs', max_iter=1000)
        X_vec = self.vectorizer.fit_transform(X)
        best_model.fit(X_vec, y)

        with open(self.model_path, 'wb') as f:
            pickle.dump((self.vectorizer, best_model), f)

        self.model = best_model

    def predict_reason_classifier(self, reason: str) -> str:
        if self.model is None:
            with open(self.model_path, 'rb') as f:
                self.vectorizer, self.model = pickle.load(f)

        vector = self.vectorizer.transform([reason])
        prediction = self.model.predict(vector)
        return prediction[0]


if __name__ == '__main__':
    classifier = ProblemClassifier()
    classifier.fit_reason_classifier()
    print(classifier.predict_reason_classifier('Сломался дроссель'))
