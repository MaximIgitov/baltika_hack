import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.metrics import roc_auc_score, f1_score
import optuna
from optuna.samplers import TPESampler
import lightgbm as lgb

# Датасет, хранящий ошибки
df = pd.read_excel("Categorical breakdowns.xlsx")

# Датасеты признаков и таргетов, генерируемые из датасета ошибок
features_data = pd.read_excel("Features_3H.xlsx")
target_data = pd.read_excel("Target_3H.xlsx")

WINDOW_SIZES = [1, 3, 7, 14, 30, 60, 180]

def insert_breakdown(reason: str, start: pd.Timestamp, end: pd.Timestamp, duration: float) -> None:
    # Генерация короткого имени на основе уже размеченных данных
    reason_short = <...>(reason)
    df = df.concat(df, pd.DataFrame({"start": start, "end": end, "reason": reason, "duration": duration, "reason_short": reason_short}))
    df.drop_duplicates()

def generate_features_and_targets(start: pd.Timestamp, end: pd.Timestamp, freq: int = 3, prediction_horizon: int = 3) -> None:
    global feature_data, target_data, df
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

def fit_model() -> None:
    