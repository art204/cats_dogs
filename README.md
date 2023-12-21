# cats_dogs
Бинарная классификация изображений: cat(класс 0) / dog(класс 1).

Как запустить проект:
1. клонировать репозиторий
2. создать новый чистый virtualenv
3. установить зависимости: `poetry install`
4. установить pre-commit: `pre-commit install`. Запустить все хуки можно с помошью команды: `pre-commit run -a`
5. данные лежат на drive.google.com.
Скачать данные: `dvc pull`. Эта команда выполняется автоматически при обучении (train.py) и инференсе (infer.py).
6. Перед выполнением `train.py` или `infer.py` добавить директорию проекта в PYTHONPATH (Для WINDOWS: `set PYTHONPATH=.`).
7. Обучение модели: `python ds_project/train.py`.
Модель сохраняется в: `models/имя-модели.ckpt`. Имя модели
задано в `conf_hydra/config_train.yaml` параметром `ckpt_name`.
Путь к данным для обучения задан в `conf_hydra/data/catdog_data.yaml` параметром `data_train`.
8. Гиперпараметры, имя модели (ckpt_name), пути к данным и другие настройки заданы в папке `conf_hydra`
в соответствующих `.yaml` файлах.
9. Для логирования используется `mlflow`. По умолчанию uri для mlflow `http://127.0.0.1:8080`
задан в `conf_hydra/mlflow/mlflow_local.yaml`. Запуск mlflow: `mlflow server --host 127.0.0.1 --port 8080`.
10. Инференс модели: `python ds_project/infer.py`. Результат сохраняется в: `data/имя-папки-с-данными-для-инференса_имя-модели.csv`
Имя папки с данными для инференса задано в `conf_hydra/data/catdog_data.yaml` параметром `data_infer`. Имя модели
задано в `conf_hydra/config_infer.yaml` параметром `ckpt_name`.
