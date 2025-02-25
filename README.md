# Проект прогнозирования сбоев оборудования

Этот проект предназначен для прогнозирования сбоев оборудования на основе исторических данных. Проект включает в себя два основных компонента: классификатор причин сбоев и предсказатель сбоев.

## Структура проекта

- **models/**: Директория для хранения обученных моделей.
  - `breakdown_model1.pkl`: Модель для прогнозирования вероятности поломки на 1, 3, 30 дней вперед.
  - `breakdown_model2.pkl`: Модель для предсказания следующей поломки.
  - `reason_classifier.pkl`: Модель для классификации причин сбоев.
  
- **resources/**: Директория для хранения данных.
  - `Categorical breakdowns (V4).xlsx`: Данные о сбоях + разметка (необязательно).
  - `Features_3H.xlsx`: Сгенерированные признаки.
  - `Target_3H.xlsx`: Целевые переменные.
  
- **pipeline/**: Основной код проекта.
  - `Pipeline.py`: Основной скрипт для генерации признаков, обучения моделей и прогнозирования.
  - `ReasonClassifier.py`: Классификатор причин сбоев.

## Установка и использование

1. **Установка зависимостей**:
   Убедитесь, что у вас установлены все необходимые зависимости. Вы можете установить их с помощью команды:
   ```bash
   pip install -r requirements.txt
Запуск проекта:

Для обучения моделей и прогнозирования сбоев выполните:

python
Copy
python pipeline/Pipeline.py
Убедитесь, что данные в директории resources/ корректны и актуальны.

Основные функции
Классификатор причин сбоев (ReasonClassifier)
fit_reason_classifier: Обучение модели классификации причин сбоев.

predict_reason_classifier: Предсказание причины сбоя на основе текстового описания.

Предсказатель сбоев (BreakdownPredictor)
insert_breakdown: Добавление нового сбоя в данные.

generate_features_and_targets: Генерация признаков и целевых переменных на основе исторических данных.

fit_model: Обучение моделей для прогнозирования сбоев.

predict: Прогнозирование вероятности сбоев на заданный временной диапазон.

Пример использования представлен в Pipeline.py

Лицензия
Этот проект распространяется под лицензией MIT. Подробности см. в файле LICENSE.
