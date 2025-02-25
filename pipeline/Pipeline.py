import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb

from openfe import OpenFE, transform

from typing import Tuple

import warnings
warnings.filterwarnings("ignore")


# Датасет, хранящий ошибки
df = pd.read_excel("Categorical breakdowns.xlsx")

# Датасеты признаков и таргетов, генерируемые из датасета ошибок
feature_data = pd.read_excel("Features_3H.xlsx")
target_data = pd.read_excel("Target_3H.xlsx")

WINDOW_SIZES = [1, 3, 7, 14, 30, 60, 180]
PREDICTION_HORIZON = 3

model1 = 0
model2 = 0

def insert_breakdown(reason: str, start: pd.Timestamp, end: pd.Timestamp, duration: float) -> None:
    # Генерация короткого имени на основе уже размеченных данных
    reason_short = <...>(reason)
    df = df.concat(df, pd.DataFrame({"start": start, "end": end, "reason": reason, "duration": duration, "reason_short": reason_short}))
    df.drop_duplicates()

def generate_features_and_targets(start: pd.Timestamp, end: pd.Timestamp, freq: int = 3, prediction_horizon: int = 3) -> None:
    global feature_data, target_data, df, PREDICTION_HORIZON
    PREDICTION_HORIZON = prediction_horizon
    freq_str = f'{freq}h'
    time_range = pd.date_range(start=start, end=end, freq='3h')
    
    # Создаем новые DataFrames для хранения признаков и таргетов
    feature_rows = []  # Список для хранения строк признаков
    target_rows = []   # Список для хранения строк таргетов
    
    for now in time_range:
        # Создаем строку для текущего времени
        new_feature_row = {'date': now}  # Заполняем временные рамки
        new_target_row = {'date': now}    # Заполняем временные рамки для таргетов
        
        for reason in df['reason_short'].unique():
        
            for window in WINDOW_SIZES:
                # Время начала окна
                start_window = now - pd.Timedelta(days=window)
                
                # Фильтруем строки по текущему reason_short и временным границам
                mask = (df['reason_short'] == reason) & (df['start'] < now) & (df['end'] > start_window)
                filtered_df = df[mask]
                    
                new_feature_row[f'{reason}_within_last_{window}_days'] = int(not filtered_df.empty)
                new_feature_row[f'{reason}_count_last_{window}_days'] = len(filtered_df)
                new_feature_row[f'{reason}_duration_sum_last_{window}_days'] = filtered_df['duration'].sum() if not filtered_df.empty else 0
                
            # Конец предсказания
            end_window = now + pd.Timedelta(days=prediction_horizon) 
            mask = (df['reason_short'] == reason) & (df['start'] > now) & (df['start'] < end_window)
            filtered_df = df[mask]
                    
            new_target_row[f'{reason}_within_next_{prediction_horizon}_days'] = int(not filtered_df.empty)
            new_target_row[f'{reason}_count_next_{prediction_horizon}_days'] = len(filtered_df)
            new_target_row[f'{reason}_duration_sum_next_{prediction_horizon}_days'] = filtered_df['duration'].sum() if not filtered_df.empty else 0
            
        new_target_row["next_breakdown"] = df[df["start"] > now].iloc[0]["reason_short"]
        
        # Добавляем новую строку в список признаков
        feature_rows.append(new_feature_row)
        
        target_rows.append(new_target_row)

    # Создаем DataFrame из списков строк
    feature_df = pd.DataFrame(feature_rows)
    target_df = pd.DataFrame(target_rows)


def fit_model(reason: str, top_breakdowns_num: int = 4) -> None:
    global feature_data, target_data, model1, model2
    top_breakdowns = target_data["next_breakdown"].value_counts().nlargest(top_breakdowns_num).index
    target_data_red = target_data[target_data["next_breakdown"].isin(top_breakdowns)]

    feature_data_red = feature_data.loc[feature_data.index.isin(target_data_red.index)]
    
    # Загрузка данных
    X = feature_data_red.drop(columns=["date"])

    # Преобразуем события в числовые метки
    event_encoder = LabelEncoder()
    y = event_encoder.fit_transform(target_data_red["next_breakdown"])

    # Разделение данных на обучающую и тестовую выборки
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    
    # Преобразование y в бинарный формат
    lb = LabelBinarizer()
    y_test_binarized = lb.transform(y_test)

    # Функция для Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 10, 50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)

        model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=239, verbose=-1)
    
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:]
        return roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=50)

    # Лучшие параметры
    best_params = study.best_params
    print(f'Best ROC-AUC: {study.best_value:.2f}')
    print(f'Best parameters: {best_params}')

    # Обучение модели с лучшими параметрами
    model1 = lgb.LGBMClassifier(**best_params, random_state=42)

    model1.fit(X_train, y_train)
    
    feature_importances = model1.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Отберем 50 самых важных признаков
    feature_importance_df_selected = feature_importance_df[:50]
    
    X = feature_data_red.drop(columns=["date"])[feature_importance_df_selected["Feature"]]

    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    
    #Генерируем новые признаки
    ofe = OpenFE()
    features = ofe.fit(data=X_train, label=y_train)
    train_x, test_x = transform(X_train, X_test, features, n_jobs=8)

    # Функция для Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)

        model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=239, verbose=-1)
    
        model.fit(train_x, y_train)
        y_pred_proba = model.predict_proba(test_x)[:]
        return roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=15)

    # Обучение модели с лучшими параметрами
    model1 = lgb.LGBMClassifier(**best_params, random_state=42)

    model1.fit(train_x, y_train)

    
    # Загрузка данных
    X = feature_data.drop(columns=["date"])

    # Преобразуем события в числовые метки
    event_encoder = LabelEncoder()
    y = event_encoder.fit_transform(target_data[f'{reason}_within_next_{PREDICTION_HORIZON}_days'])
    
    train_size = int(0.8 * len(X))
    X_train, X_test, y_train, y_test = X[:train_size], X[train_size], y[:train_size], y[train_size:]
    
    from sklearn.metrics import f1_score

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

    model2 = lgb.LGBMClassifier(**best_params, random_state=42)

    model2.fit(X_train, y_train)
    
    feature_importances = model2.feature_importances_

    feature_importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Отберем 50 самых важных признаков
    feature_importance_df_selected = feature_importance_df[:50]
    
    X = feature_data.drop(columns=["date"])[feature_importance_df_selected["Feature"]]

    X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
    
    #Генерируем новые признаки
    ofe = OpenFE()
    features = ofe.fit(data=X_train, label=y_train)
    train_x, test_x = transform(X_train, X_test, features, n_jobs=8)

    # Функция для Optuna
    def objective(trial):
        n_estimators = trial.suggest_int('n_estimators', 50, 200)
        max_depth = trial.suggest_int('max_depth', 5, 50)
        learning_rate = trial.suggest_float('learning_rate', 0.01, 0.1)

        model = lgb.LGBMClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=239, verbose=-1)
    
        model.fit(train_x, y_train)
        y_pred_proba = model.predict_proba(test_x)[:]
        return roc_auc_score(y_test_binarized, y_pred_proba, multi_class='ovr')

    study = optuna.create_study(direction='maximize', sampler=TPESampler())
    study.optimize(objective, n_trials=15)

    # Обучение модели с лучшими параметрами
    model2 = lgb.LGBMClassifier(**best_params, random_state=42)

    model2.fit(train_x, y_train)
    
def predict(time_range: pd.core.indexes.datetimes.DatetimeIndex) -> Tuple[np.array]:
    # Создаем новые DataFrames для хранения признаков
    feature_rows = []  # Список для хранения строк признаков
    
    for now in time_range:
        # Создаем строку для текущего времени
        new_feature_row = {'date': now}  # Заполняем временные рамки
        
        for reason in df['reason_short'].unique():
        
            for window in WINDOW_SIZES:
                # Время начала окна
                start_window = now - pd.Timedelta(days=window)
                
                # Фильтруем строки по текущему reason_short и временным границам
                mask = (df['reason_short'] == reason) & (df['start'] < now) & (df['end'] > start_window)
                filtered_df = df[mask]
                    
                new_feature_row[f'{reason}_within_last_{window}_days'] = int(not filtered_df.empty)
                new_feature_row[f'{reason}_count_last_{window}_days'] = len(filtered_df)
                new_feature_row[f'{reason}_duration_sum_last_{window}_days'] = filtered_df['duration'].sum() if not filtered_df.empty else 0
                
        # Добавляем новую строку в список признаков
        feature_rows.append(new_feature_row)

    # Создаем DataFrame из списков строк
    X = pd.DataFrame(feature_rows)
    
    # Возвращаем вероятности разных ошибок быть следующими
    # и вероятность возникновения определенной ошибки на определенный горизонт
    return (model1.predict_proba(X), model2.predict_proba(X))
    