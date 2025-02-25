# import sys
# sys.path.append(r'D:\repositories\baltika_hack')

import numpy as np
import pandas as pd
import os
import pickle
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb
from openfe import OpenFE, transform
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings

from ReasonClassifier import ReasonClassifier
from ReasonClassifier import ReasonClassifier
import warnings
warnings.filterwarnings("ignore")


class BreakdownPredictor:
    def __init__(self, model1_path: str = 'pipeline/models/breakdown_model1.pkl', model2_path: str = 'pipeline/models/breakdown_model2.pkl'):
        self.model1_path = model1_path
        self.model2_path = model2_path
        self.model1 = None
        self.model2 = None
        self.WINDOW_SIZES = [1, 3, 7, 14, 30, 60, 180]
        self.PREDICTION_HORIZON = 3

    def insert_breakdown(self, reason: str, start: pd.Timestamp, end: pd.Timestamp, duration: float) -> None:
        reason_classifier = ReasonClassifier()
        reason_short = reason_classifier.predict_reason_classifier(reason)
        df = pd.read_excel("pipeline/resources/Categorical breakdowns (V4).xlsx")
        new_row = pd.DataFrame({"start": [start], "end": [end], "reason": [reason], "duration": [duration], "reason_short": [reason_short]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_excel("pipeline/resources/Categorical breakdowns (V4).xlsx", index=False)

    def generate_features_and_targets(self, start: pd.Timestamp, end: pd.Timestamp, freq: int = 3, prediction_horizon: int = 3) -> None:
        df = pd.read_excel("pipeline/resources/Categorical breakdowns (V4).xlsx")
        time_range = pd.date_range(start=start, end=end, freq=f'{freq}h')
        feature_rows = []
        target_rows = []

        for now in time_range:
            new_feature_row = {'date': now}
            new_target_row = {'date': now}

            for reason in df['reason_short'].unique():
                for window in self.WINDOW_SIZES:
                    start_window = now - pd.Timedelta(days=window)
                    mask = (df['reason_short'] == reason) & (df['start'] < now) & (df['end'] > start_window)
                    filtered_df = df[mask]

                    new_feature_row[f'{reason}_within_last_{window}_days'] = int(not filtered_df.empty)
                    new_feature_row[f'{reason}_count_last_{window}_days'] = len(filtered_df)
                    new_feature_row[f'{reason}_duration_sum_last_{window}_days'] = filtered_df['duration'].sum() if not filtered_df.empty else 0

                end_window = now + pd.Timedelta(days=prediction_horizon)
                mask = (df['reason_short'] == reason) & (df['start'] > now) & (df['start'] < end_window)
                filtered_df = df[mask]

                new_target_row[f'{reason}_within_next_{prediction_horizon}_days'] = int(not filtered_df.empty)
                new_target_row[f'{reason}_count_next_{prediction_horizon}_days'] = len(filtered_df)
                new_target_row[f'{reason}_duration_sum_next_{prediction_horizon}_days'] = filtered_df['duration'].sum() if not filtered_df.empty else 0

            new_target_row["next_breakdown"] = df[df["start"] > now].iloc[0]["reason_short"]
            feature_rows.append(new_feature_row)
            target_rows.append(new_target_row)

        feature_df = pd.DataFrame(feature_rows)
        target_df = pd.DataFrame(target_rows)
        feature_df.to_excel("pipeline/resources/Features_3H.xlsx", index=False)
        target_df.to_excel("pipeline/resources/Target_3H.xlsx", index=False)

    def fit_model(self, reason: str, top_breakdowns_num: int = 4) -> None:
        feature_data = pd.read_excel("pipeline/resources/Features_3H.xlsx")
        target_data = pd.read_excel("pipeline/resources/Target_3H.xlsx")
        top_breakdowns = target_data["next_breakdown"].value_counts().nlargest(top_breakdowns_num).index
        target_data_red = target_data[target_data["next_breakdown"].isin(top_breakdowns)]
        feature_data_red = feature_data.loc[feature_data.index.isin(target_data_red.index)]

        X = feature_data_red.drop(columns=["date"])
        event_encoder = LabelEncoder()
        y = event_encoder.fit_transform(target_data_red["next_breakdown"])

        train_size = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

        print("Unique classes in y_train:", np.unique(y_train))
        print("Unique classes in y_test:", np.unique(y_test))

        lb = LabelBinarizer()
        y_test_binarized = lb.fit_transform(y_test)

        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 10, 50)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)

            model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=239, verbose=-1)
            model.fit(X_train, y_train)
            y_pred_proba = model.predict_proba(X_test)

            # Subset y_pred_proba to match classes in y_test
            model_classes = model.classes_
            test_classes = np.unique(y_test)
            class_indices = [np.where(model_classes == cls)[0][0] for cls in test_classes]
            y_pred_proba_subset = y_pred_proba[:, class_indices]

            return roc_auc_score(y_test_binarized, y_pred_proba_subset, multi_class='ovr')

        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=50)

        best_params = study.best_params
        self.model1 = lgb.LGBMClassifier(**best_params, random_state=42)
        self.model1.fit(X_train, y_train)

        # Second model (binary classification) remains mostly unchanged
        column_name = f'{reason}_within_next_{self.PREDICTION_HORIZON}_days'
        if column_name not in target_data.columns:
            raise ValueError(f"Column '{column_name}' not found in target_data. Please check the data generation process.")

        X = feature_data.drop(columns=["date"])
        y = event_encoder.fit_transform(target_data[column_name])
        train_size = int(0.8 * len(X))
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

        def objective(trial):
            n_estimators = trial.suggest_int('n_estimators', 50, 200)
            max_depth = trial.suggest_int('max_depth', 10, 50)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)

            model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=239, verbose=-1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            return f1_score(y_test, y_pred, average='weighted')

        study = optuna.create_study(direction='maximize', sampler=TPESampler())
        study.optimize(objective, n_trials=50)

        self.model2 = lgb.LGBMClassifier(**best_params, random_state=42)
        self.model2.fit(X_train, y_train)

        with open(self.model1_path, 'wb') as f:
            pickle.dump(self.model1, f)
        with open(self.model2_path, 'wb') as f:
            pickle.dump(self.model2, f)

    def predict(self, time_range: pd.core.indexes.datetimes.DatetimeIndex) -> Tuple[np.array]:
        df = pd.read_excel("pipeline/resources/Categorical breakdowns (V4).xlsx")
        feature_rows = []

        for now in time_range:
            new_feature_row = {'date': now}

            for reason in df['reason_short'].unique():
                for window in self.WINDOW_SIZES:
                    start_window = now - pd.Timedelta(days=window)
                    mask = (df['reason_short'] == reason) & (df['start'] < now) & (df['end'] > start_window)
                    filtered_df = df[mask]

                    new_feature_row[f'{reason}_within_last_{window}_days'] = int(not filtered_df.empty)
                    new_feature_row[f'{reason}_count_last_{window}_days'] = len(filtered_df)
                    new_feature_row[f'{reason}_duration_sum_last_{window}_days'] = filtered_df['duration'].sum() if not filtered_df.empty else 0

            feature_rows.append(new_feature_row)

        X = pd.DataFrame(feature_rows).drop(columns=["date"])

        if self.model1 is None:
            with open(self.model1_path, 'rb') as f:
                self.model1 = pickle.load(f)
        if self.model2 is None:
            with open(self.model2_path, 'rb') as f:
                self.model2 = pickle.load(f)

        return (self.model1.predict_proba(X), self.model2.predict_proba(X))


if __name__ == '__main__':
    reason_classifier = ReasonClassifier()
    reason_classifier.fit_reason_classifier()

    breakdown_predictor = BreakdownPredictor()
    breakdown_predictor.generate_features_and_targets(pd.Timestamp('2023-01-01'), pd.Timestamp('2023-12-31'))
    breakdown_predictor.fit_model('Замена двух дросселей')
    predictions = breakdown_predictor.predict(pd.date_range(start=pd.Timestamp('2024-01-01'), periods=10, freq='3h'))
    print(predictions)
