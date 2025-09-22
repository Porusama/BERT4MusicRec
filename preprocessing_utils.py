from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import joblib
import numpy as np
import os
from config import CURRENT_DIR


__ENCODERS_DIR = "preprocessing encoders"


COLUMS_FOR_SCALING = [
    'danceability', 'energy', 'loudness',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo', "duration_ms"
]


ENCODER_PATHS = {
    'songs_le': os.path.join(CURRENT_DIR, __ENCODERS_DIR, "label_encoder_songs.pkl"),
    'albums_le': os.path.join(CURRENT_DIR, __ENCODERS_DIR, "label_encoder_albums.pkl"),
    'artists_le': os.path.join(CURRENT_DIR, __ENCODERS_DIR, "label_encoder_artists.pkl"),
    'standart_scaler': os.path.join(CURRENT_DIR, __ENCODERS_DIR, "standart_scaler.pkl"),
    'one_hot': os.path.join(CURRENT_DIR, __ENCODERS_DIR, "one_hot_encoder.pkl")
}


def get_songs_le():
    return joblib.load(ENCODER_PATHS["songs_le"])

    
def preprocess_data(df, mode=False):
    ''' Производит предобработку датафрейма
        mode=True  - первичная предобработка данных с сохранением моделей для предобработки на диск
        mode=False - предобработка данных для инференса с загрузкой сохраненных моделей с диска
    '''

    df['duration_ms'] = np.log1p(df['duration_ms'])

    if mode:
        # Создает и обучает модели для предобработки
        standart_scaler = StandardScaler().fit(df[COLUMS_FOR_SCALING])
        one_hot_encoder = OneHotEncoder(sparse_output=False).fit(df[['time_signature']])
        le_songs        = LabelEncoder().fit(df['song_id'])
        le_albums       = LabelEncoder().fit(df['album_id'])
        le_artists      = LabelEncoder().fit(df['artist_id'])
    else:
        # Загружает с диска
        standart_scaler = joblib.load(ENCODER_PATHS["standart_scaler"])
        one_hot_encoder = joblib.load(ENCODER_PATHS["one_hot"])
        le_songs = joblib.load(ENCODER_PATHS["songs_le"])
        le_albums = joblib.load(ENCODER_PATHS["albums_le"])
        le_artists = joblib.load(ENCODER_PATHS["artists_le"])

    # Нормализация числовых колонок
    df[COLUMS_FOR_SCALING] = standart_scaler.transform(df[COLUMS_FOR_SCALING])
    
    # Унитарный код
    time_signature_ohe = one_hot_encoder.transform(df[['time_signature']])
    ts_cols = [f"time_signature_{int(c)}" for c in one_hot_encoder.categories_[0]]
    df[ts_cols] = time_signature_ohe

    # Кодирование метками
    df['song_id'] = le_songs.transform(df['song_id']) + 2 
    df['album_id'] = le_albums.transform(df['album_id']) + 1
    df['artist_id'] = le_artists.transform(df['artist_id']) + 1

    df['explicit'] = df['explicit'].astype(int)
    
    # Циклическое кодирование
    df['key_sin'] = np.sin(2 * np.pi * df['key'] / 12)
    df['key_cos'] = np.cos(2 * np.pi * df['key'] / 12)

    df.drop(['pos', 'key', 'time_signature'], axis=1, inplace=True)

    if mode:
        joblib.dump(standart_scaler, ENCODER_PATHS['standart_scaler'])
        joblib.dump(one_hot_encoder, ENCODER_PATHS['one_hot'])
        joblib.dump(le_songs,        ENCODER_PATHS['songs_le'])
        joblib.dump(le_albums,       ENCODER_PATHS['albums_le'])
        joblib.dump(le_artists,      ENCODER_PATHS['artists_le'])