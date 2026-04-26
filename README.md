# Credit Card Fraud Detection: Predictive Models

Проект посвящен задаче обнаружения мошеннических операций по банковским картам на основе анонимизированного датасета транзакций. Основная цель — сравнить несколько моделей машинного обучения и оценить, насколько хорошо они определяют fraud-транзакции в условиях сильного дисбаланса классов.

## Что внутри

В ноутбуке последовательно выполнены основные этапы ML-анализа:

- загрузка библиотек и данных;
- первичная проверка датасета;
- анализ пропусков и дисбаланса целевой переменной;
- exploratory data analysis по времени, сумме транзакций, корреляциям и распределениям признаков;
- обучение и сравнение моделей RandomForest, AdaBoost, CatBoost, XGBoost и LightGBM;
- оценка качества моделей по ROC-AUC;
- итоговые выводы по результатам экспериментов.

## Данные

Используется датасет `creditcard.csv` с транзакциями по банковским картам. Целевая переменная — `Class`, где `0` означает обычную транзакцию, а `1` — мошенническую. Датасет сильно несбалансирован: fraud-операции составляют небольшую долю от всех наблюдений.

Для запуска ноутбука локально нужно скачать датасет и указать корректный путь к файлу `creditcard.csv` в блоке чтения данных.

## Модели

В проекте сравниваются несколько алгоритмов классификации:

- RandomForestClassifier;
- AdaBoostClassifier;
- CatBoostClassifier;
- XGBoost;
- LightGBM.

Качество моделей оценивается с помощью ROC-AUC, так как при сильном дисбалансе классов обычная accuracy может быть недостаточно информативной.

## Основные результаты

Лучшие результаты показали бустинговые модели. XGBoost достиг AUC около `0.974` на test set, LightGBM показал AUC около `0.946` на test set, а при cross-validation — около `0.93`. RandomForest, AdaBoost и CatBoost также были протестированы, но показали более низкие значения AUC в рамках данного эксперимента.

## Стек

- Python;
- pandas, numpy;
- matplotlib, seaborn, plotly;
- scikit-learn;
- CatBoost;
- XGBoost;
- LightGBM.

## Как запустить

1. Установить зависимости из используемого Python-окружения.
2. Скачать датасет `creditcard.csv`.
3. Указать путь к файлу в ноутбуке.
4. Запустить ячейки последовательно от блока `Load packages` до `Conclusions`.

## Краткое описание для GitHub

Machine learning project for credit card fraud detection on an imbalanced transaction dataset. The notebook includes EDA, class imbalance analysis, feature exploration, and comparison of RandomForest, AdaBoost, CatBoost, XGBoost, and LightGBM models using ROC-AUC as the main quality metric.
