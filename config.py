import os

CURRENT_DIR  = os.getcwd()

# Метаданные демонстрационного датасета (./demo data/sample.csv)
demo_sample_meta = {
    'playlists_amo':        3000,
    'unique_songs':         11540 + 2, # for [PAD] = 0, [MASK] = 1
    'unique_artists':       2343  + 1, # for [PAD]  and [MASK] = 0
    'unique_albums':        3916  + 1, # for [PAD]  and [MASK] = 0
    'numeric_features_amo': 21,
    'num_epochs': 5,
    'demo_sequence_path': 'demo data\\sequence.csv',
    'demo_dataset_path': 'demo data\\sample.csv',
    'demo_names_path': 'demo data\\names.csv'
}

# Конфигурация модели
config = {
    # Метаданные датасета (определяют размеры таблиц эмбеддингов)
    'playlists_amo':        305087,
    'unique_songs':         116291 + 2, # for [PAD] = 0, [MASK] = 1
    'unique_artists':       12560  + 1, # for [PAD]  and [MASK] = 0
    'unique_albums':        22151  + 1, # for [PAD]  and [MASK] = 0
    'numeric_features_amo': 21,

    # Относительный путь к датасету
    'dataset_path': '',

    # Гиперпараметры модели
    'seq_len': 32,              # Длина последовательности прослушивания
    'cloze_probability': 0.25,  # Вероятность cloze маскировки
    'd_songs': 64,              # Размерность эмбеддинг вектора для 1) айди песен
    'd_albums': 32,             #                                   2) айди альбомов
    'd_artists': 32,            #                                   3) айди артистов
    'd_ff': 512,                # Размерность скрытого слоя feed forward блока
    'N': 3,                     # Количество кодировщиков (глубина модели)
    'heads': 2,                 # Количество голов самовнимания (кратные d_model = d_songs + d_album + d_artists)
    'dropout_prob': 0.1,        # Вероятность dropout

    # Гиперпараметры обучения
    'train_size': 0.9,          # Объем обучающей выборки
    'batch_size': 16,           # Размер батча
    'num_epochs': 20,           # Количество эпоъ
    'lr': 1e-5,                 # Скорость обучения

    # Определение путей
    'model_folder': 'model\\model_weights',  # Папка с прогрессом обучения
    'model_basename': 'model_epoch_',       # Название модели
    'writer_dict': 'model\\logs\\BERTmodel',  # Папка для логирования валидационных метрик
    'epoch_to_preload': 0,                  # Эпоха для предзагрузки (если равно 0, то предзагрзука модели не происходит)
    'epoch_to_load': 5
}

def get_models_weights_file_path(config, epoch: int):
    model_folder = config['model_folder']
    model_filename = f"{config['model_basename']}{epoch:02d}.pt"
    return os.path.join(CURRENT_DIR, model_folder, model_filename)