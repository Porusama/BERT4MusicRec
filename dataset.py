import torch
import pandas as pd
from torch.utils.data import Dataset
import random
import numpy as np


def dataset_proper_load(path:str):
    ''' Функция загрузки датасета из csv файла
    '''
    df = pd.read_csv(path)
    df = df.groupby('playlist_id')
    return df


class PlaylistDataset(Dataset):
    def __init__(self, seq_len:int, data: pd.core.groupby.DataFrameGroupBy, cloze_prob: float=0.25, mode: bool=True):
        ''' mode - Тип датасета
            mode=True -  обучающая выборка (cloze)
            mode=False - тестовая выборка  (v(-1) = [MASK])
        '''
        self.mode=mode

        self.seq_len = seq_len

        self.data = data

        # Определяем колонки характеристик
        self.num_f_cols = ['explicit',          'mode',             'disc_number',      'track_number', # Целые колонки
                           'danceability',      'energy',           'loudness',                         # Дробные колонки
                           'speechiness',       'acousticness',     'instrumentalness',
                           'liveness',          'valence',          'tempo',
                           'duration_ms',       'key_sin',          'key_cos',
                           'time_signature_0',  'time_signature_1', 'time_signature_3',
                           'time_signature_4',  'time_signature_5']
        
        self.cloze_prob = cloze_prob

        # Задаем индексы маски и паддинга
        self.pad_index  = 0
        self.mask_index = 1

        # Ниже буферы для быстрого задания выходных тензоров

        # Вектора длины seq заполненнные [MASK] и [PAD] индексами
        self.buff_pad_song_ids  = torch.full((self.seq_len,), self.pad_index,  dtype=torch.int64)
        self.buff_mask_song_ids = torch.full((self.seq_len,), self.mask_index, dtype=torch.int64)

        # Матрица нулей для числовых характеристик (seq, num_features)
        self.buff_num_features = torch.zeros((self.seq_len, len(self.num_f_cols)), dtype=torch.float)
        
        # Вектор нулей для альбомов и артистов
        self.buff_album_artists_ids = torch.zeros((self.seq_len,), dtype=torch.int)

        # Вектор единиц для паддинг матрицы
        self.buff_padding_mask = torch.ones(self.seq_len, dtype=torch.bool)

        # Вектор элементов игнорирования для кросс энтропии
        self.buff_label_mask = torch.full((self.seq_len, ), -100, dtype=torch.int64)


    def __len__(self):
        return len(self.data)
    

    def __getitem__(self, index):        
        # Получение плейлиста
        playlist = self.data[index]
        plst_len = len(playlist)

        # Случайный срез для длинных плейлистов
        if plst_len > self.seq_len:
            start_of_cut = random.randint(0, plst_len - self.seq_len)
            playlist = playlist[start_of_cut:start_of_cut + self.seq_len]
            plst_len = self.seq_len

        # Добавялем индекс последнего элемента для маскировки
        indices_to_mask = [plst_len - 1]
        
        # Формируем массив маскируемых индексов
        if self.mode:
            # Выбираем индексы для замены
            num_to_replace = int((plst_len - 1) * self.cloze_prob)
            indices_to_mask.extend(np.random.choice(plst_len, num_to_replace, replace=False))

        # Формируем массив немаскируемых индексов
        indices_not_to_mask = set(range(0, plst_len))
        indices_not_to_mask = [x for x in indices_not_to_mask if x not in indices_to_mask]

        # Получаем идентификаторы песен в альбоме
        out_song_ids = self.buff_pad_song_ids.clone()
        out_song_ids[:plst_len] = torch.tensor(playlist['song_id'].to_list(), dtype=torch.int64)

        # Сохраняем маскируемые айди песен в массив игнорирования (так они не будут игнорироваться) 
        out_label = self.buff_label_mask.clone()
        out_label[indices_to_mask] = out_song_ids[indices_to_mask]

        # Формируем маску для слоя самовнимания
        out_padding_mask = self.buff_padding_mask.clone()
        out_padding_mask[plst_len:] = 0

        # Маскируем необходимые айди песен
        out_song_ids[indices_to_mask] = self.buff_mask_song_ids[indices_to_mask]

        # Получаем немаскируемые наборы числовых характеристик
        out_num_features = self.buff_num_features.clone()
        out_num_features[indices_not_to_mask] = torch.tensor(playlist[self.num_f_cols].values.tolist(), dtype=torch.float)[indices_not_to_mask]

        # Получаем немаскируемые значения альбомов
        out_album_ids = self.buff_album_artists_ids.clone()
        out_album_ids[indices_not_to_mask] = torch.tensor(playlist['album_id'].to_list(), dtype=torch.int)[indices_not_to_mask]

        # Получаем немаскируемые значения альбомов
        out_artist_ids = self.buff_album_artists_ids.clone()
        out_artist_ids[indices_not_to_mask] = torch.tensor(playlist['artist_id'].to_list(), dtype=torch.int)[indices_not_to_mask]

        return {
            "song_ids":         out_song_ids,                       # (seq)
            "numeric_features": out_num_features,                   # (seq * num_features) 
            "album_ids":        out_album_ids,                      # (seq)
            "artist_ids":       out_artist_ids,                     # (seq)
        },  out_padding_mask.unsqueeze(0).unsqueeze(0),   out_label # (1(batch), 1(head), seq) и (seq)