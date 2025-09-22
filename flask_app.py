import os
from dotenv import load_dotenv

import pandas as pd

from preprocessing_utils import preprocess_data, get_songs_le
from train import get_model
from config import config, get_models_weights_file_path, CURRENT_DIR

import torch

# Буферы для формирования последовательности
# Определяем колонки характеристик
num_f_cols =   ['explicit',          'mode',             'disc_number',      'track_number', # Целые колонки
                'danceability',      'energy',           'loudness',                         # Дробные колонки
                'speechiness',       'acousticness',     'instrumentalness',
                'liveness',          'valence',          'tempo',
                'duration_ms',       'key_sin',          'key_cos',
                'time_signature_0',  'time_signature_1', 'time_signature_3',
                'time_signature_4',  'time_signature_5']

# Задаем индексы маски и паддинга
pad_index  = 0
mask_index = 1

# Ниже буферы для быстрого задания выходных тензоров

# Вектора длины seq заполненнные [MASK] и [PAD] индексами
buff_pad_song_ids  = torch.full((config['seq_len'],), pad_index,  dtype=torch.int64)
buff_mask_song_ids = torch.full((config['seq_len'],), mask_index, dtype=torch.int64)

# Матрица нулей для числовых характеристик (seq, num_features)
buff_num_features = torch.zeros((config['seq_len'], len(num_f_cols)), dtype=torch.float)

# Вектор нулей для альбомов и артистов
buff_album_artists_ids = torch.zeros((config['seq_len'],), dtype=torch.int)

# Вектор единиц для паддинг матрицы
buff_padding_mask = torch.ones(config['seq_len'], dtype=torch.bool)


def load_model(config, device, epoch: int = 0):
    ''' Загружает модель в память
    '''

    model = get_model(config)
    model.to(device)
    model.eval()
    model_filename = get_models_weights_file_path(config, epoch)
    state = torch.load(model_filename)
    model.load_state_dict(state['model'])

    return model


def form_inference_sequence(df, device):
    ''' Создает последовательность определенного формата для обработки нейросетью
    '''

    plst_len = len(df)

    # Срез для длинных плейлистов
    if plst_len > config['seq_len'] - 1:
        len_diff = plst_len - config['seq_len']
        df = df[len_diff + 1:]
        plst_len = config['seq_len'] - 1

    indices_to_mask = [plst_len]
    indices_not_to_mask = set(range(0, plst_len))
    indices_not_to_mask = [x for x in indices_not_to_mask if x not in indices_to_mask]

    # Получаем идентификаторы песен в альбоме
    out_song_ids = buff_pad_song_ids.clone()
    out_song_ids[:plst_len] = torch.tensor(df['song_id'].to_list(), dtype=torch.int64)

    # Формируем маску для слоя самовнимания
    out_padding_mask = buff_padding_mask.clone()
    out_padding_mask[plst_len:] = 0

    # Маскируем необходимые айди песен
    out_song_ids[indices_to_mask] = buff_mask_song_ids[indices_to_mask]

    # Получаем немаскируемые наборы числовых характеристик
    out_num_features = buff_num_features.clone()
    out_num_features[indices_not_to_mask] = torch.tensor(df[num_f_cols].values.tolist(), dtype=torch.float)[indices_not_to_mask]

    # Получаем немаскируемые значения альбомов
    out_album_ids = buff_album_artists_ids.clone()
    out_album_ids[indices_not_to_mask] = torch.tensor(df['album_id'].to_list(), dtype=torch.int)[indices_not_to_mask]

    # Получаем немаскируемые значения альбомов
    out_artist_ids = buff_album_artists_ids.clone()
    out_artist_ids[indices_not_to_mask] = torch.tensor(df['artist_id'].to_list(), dtype=torch.int)[indices_not_to_mask]

    return {
        "song_ids":         out_song_ids.to(device),                       # (seq)
        "numeric_features": out_num_features.to(device),                   # (seq * num_features) 
        "album_ids":        out_album_ids.to(device),                      # (seq)
        "artist_ids":       out_artist_ids.to(device),                     # (seq)
        "mask":             out_padding_mask.unsqueeze(0).unsqueeze(0).to(device),
        "masked_id":        plst_len
    }


def generate_recommendations(model, inference_seq):
    ''' Формирует массив рекомендаций
    '''

    # Прямой проход через слои-кодировщики нейросети
    encoding = model.encode(
        inference_seq['song_ids'],
        inference_seq['album_ids'],
        inference_seq['artist_ids'],
        inference_seq['numeric_features'],
        inference_seq['mask']
    )

    # Прямой проход через слой-проекцию нейросети
    projection = model.project(encoding)

    # Взятие топ-k рекомендаций
    topk = torch.topk(projection[0, inference_seq['masked_id'], :], k=10)
    return topk.indices.to(torch.device('cpu')).tolist()
    

def process_sql_request(playlist_id, conn, model, device):
    ''' Обрабатывает запрос СУБД на генерацию рекомендаций
    '''

    # Чтение запроса к БД
    query = open(
        os.path.join(
            CURRENT_DIR, 
            'sql queries\\get_certain_playlist.sql'
        )
    ).read()

    # Выполнение запроса к БД
    df = pd.read_sql_query(query, conn, params=(playlist_id, ))

    # Предобработка полученных данных
    preprocess_data(df)

    # Генерация рекомендаций
    recommendations = get_songs_le().inverse_transform(
        generate_recommendations(
            model, 
            form_inference_sequence(df, device)
        )
    ).tolist()

    # Запись рекомендаций в базу данных
    with conn:
        with conn.cursor() as cur:
            query = open(
                os.path.join(
                    CURRENT_DIR, 
                    'sql queries\\insert_recommendations.sql'
                )
            ).read()
            cur.execute(query, (playlist_id, recommendations))
    

if __name__ == '__main__':
    from flask import Flask, request, jsonify
    import psycopg2 as sql_int
    import threading
    
    # Приложение
    app = Flask(__name__)

    # Подключение к postgreSQL
    load_dotenv()

    sql_connection = sql_int.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        host=os.getenv("POSTGRES_HOST"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT")
    )

    # Устройство хранения модели
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(config, device, config['epoch_to_load'])
    
    @app.route('/recommend', methods=['POST'])
    def recommend_endpoint():
        data = request.get_json(force=True)
        playlist_id = data.get("playlist_id")

        if playlist_id is None:
            return jsonify({"error": "playlist_id is required"}), 400

        thread = threading.Thread(target=process_sql_request, args=(playlist_id, sql_connection, model, device, ))
        thread.daemon = True
        thread.start()

        return jsonify({
            "playlist_id": playlist_id,
            'status': 'accepted'
        }), 202
    
    app.run(host='0.0.0.0', port=4517)