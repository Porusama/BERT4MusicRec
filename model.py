import torch
import torch.nn as nn
import math

class CustomEmbiggingLayer(nn.Module):
    def __init__(self, cfg: dict):
        ''' Слой-конкатенация всех эмбеддингов.
            Словарь cfg должен содержать:
            - num_songs, num_albums, num_artists, num_features
            - d_song, d_album, d_artist
        
        '''
        super().__init__()
        # Определяем итоговую размерность
        self.d_model = cfg["d_song"] + cfg["d_album"] + cfg["d_artist"]

        # Категориальные эмбеддинги
        self.embeds = nn.ModuleDict({
            'song':    nn.Embedding(cfg["num_songs"],   cfg["d_song"]),
            'album':   nn.Embedding(cfg["num_albums"],  cfg["d_album"]),
            'artist':  nn.Embedding(cfg["num_artists"], cfg["d_artist"]),
            'numeric': nn.Linear(cfg['num_features'], self.d_model)
        })

        self.norm = LayerNormalization(self.d_model)

    def forward(self, song_ids, album_ids, artist_ids, num_features):
        # Отдельные эмбеддинги
        e_song   = self.embeds['song'](song_ids)        # (B, L, d_song)
        e_album  = self.embeds['album'](album_ids)      # (B, L, d_album)
        e_artist = self.embeds['artist'](artist_ids)    # (B, L, d_artist)
        e_num    = self.embeds['numeric'](num_features)   # (B, L, d_model)

        x = torch.cat([e_song, e_album, e_artist], dim=-1)
        x *= math.sqrt(self.d_model)

        return self.norm(x + e_num)
    
class PositionalEncoding(nn.Module):
    ''' Слой для кодирования информации о позиции элемента в последовательности
        вне зависимости от длины последовательности
    '''
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # Создаем матрицу размера (L, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Создаем вектор размера (L)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)

        # Создаем вектор размера (d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model / 2)

        # Применяем синус для четных индексов
        pe[:, 0::2] = torch.sin(position * div_term)                            # sin(position * (10000 ** (2i / d_model))

        # Применяем косинус для нечетных индексов
        pe[:, 1::2] = torch.cos(position * div_term)[:, :pe[:,1::2].shape[1]]   # cos(position * (10000 ** (2i / d_model))

        # Добавляем измерение батча к закодированным позициям
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        # Register the positional encoding as a buffer
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) # (B, L, d_model)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps:float=10 ** -6) -> None:
        ''' Стандартизация данных по формуле
            x' = x - mean(x)/(std(x) + eps),
            где eps - очень маленькое число 
                      для предотвращения деления на ноль
        '''
        super().__init__()
        self.eps = eps
        # alpha и beta обучаемые параметры
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x):
        # x: (B, L, d_model)
        mean = x.mean(dim = -1, keepdim = True) # (B, L, 1)
        std = x.std(dim = -1, keepdim = True)   # (B, L, 1)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias
    
class FeedForwardBlock(nn.Module):
    ''' Выполняет
        по формуле FFN(x) = max(0, x*W1 + b1)*W2 + b2
    '''

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) # W1 и b1
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model) # W2 и b2

    def forward(self, x):
        # (B, L, d_model) --> (B, L, d_ff) --> (B, L, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, dropout: float, h: int=5) -> None:
        ''' Имплементация оригинального Multi-Head Attention
            из статьи "Attention is all you need"
        '''
        super().__init__()
        self.d_model = d_model
        # Задаем количество "голов"
        self.h = h
        # Make sure d_model is divisible by h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h # Dimension of vector seen by each head
        self.w_q = nn.Linear(d_model, d_model, bias=False) # Wq
        self.w_k = nn.Linear(d_model, d_model, bias=False) # Wk
        self.w_v = nn.Linear(d_model, d_model, bias=False) # Wv
        self.w_o = nn.Linear(d_model, d_model, bias=False) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        ''' Attention(Q, K, V) = softmax(Q*K.T/sqrt(d_k))*V
        '''
        d_k = query.shape[-1]

        # (B, h, L, d_k) --(.T)--> (B, h, d_k, L)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Заполняем маскированные элементы очень маленькими значениями, эмулируя -∞
            attention_scores.masked_fill_(mask == False, -1e9)
        
        attention_scores = attention_scores.softmax(dim=-1) # (B, h, L, L) # Применяем софтмакс
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # (B, h, L, L) --> (B, h, L, d_k)
        # Возвращаем значения внимания для визуализации
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (B, L, d_model) --> (B, L, d_model)
        key = self.w_k(k)   # (B, L, d_model) --> (B, L, d_model)
        value = self.w_v(v) # (B, L, d_model) --> (B, L, d_model)

        # (B, L, d_model) --(.view)--> (B, L, h, d_k) --(.T)--> (B, h, L, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Вычисляем внимание
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Конкатенируем результаты всех голов к единой матрице
        # (batch, h, seq_len, d_k) --(.T)--> (batch, seq_len, h, d_k) --(.view)--> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Умножаем на Wo
        # (B, L, d_model) --> (B, L, d_model)  
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    ''' Реализует сквозное соединение между элементами
    '''
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(d_model)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
        
class EncoderBlock(nn.Module):
    ''' Блок-кодировщик
    '''
    def __init__(self,
                 d_model: int,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout: float):
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(d_model, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    
class Encoder(nn.Module):
    ''' Кодировщик, составленный из последовательно выполняемых
        блоков-кодировщиков
    '''
    def __init__(self, d_model: int, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
    
class ProjectionLayer(nn.Module):
    ''' Слой проецирущий выход кодировщика в массив вероятностей длины vocab_size
    '''
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x) -> None:
        # (B, L, d_model) --> (B, L, vocab_size)
        return torch.softmax(self.proj(x), dim=-1)
    
class BERT4MusicRecUltimate(nn.Module):
    ''' Итоговая модель генерации рекомендаций
        для последовательностей прослушивания
    '''
    def __init__(self,
                 encoder: Encoder,
                 src_embed: CustomEmbiggingLayer,
                 src_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer):
        super().__init__()
        self.encoder = encoder
        self.src_embed = src_embed
        self.src_pos = src_pos
        self.projection_layer = projection_layer

    def encode(self, song_ids, album_ids, artist_ids, num_features, mask):
        # (B, L, d_model)
        src = self.src_embed(song_ids, album_ids, artist_ids, num_features)
        src = self.src_pos(src)
        return self.encoder(src, mask)
    
    def project(self, x):
        # (B, L, vocab_size)
        return self.projection_layer(x)
    

def build_BERT4MusicRecUltimate_model(songs_vocab_size:     int,    albums_vocab_size:  int,
                                      artists_vocab_size:   int,    num_features:       int,
                                      d_song:               int,    d_album:            int, 
                                      d_artist:             int,    seq_len:            int,
                                      d_ff:                 int,    N:                  int=1,
                                      heads_num:            int=1,  dropout:            float=0.1) -> BERT4MusicRecUltimate:

    # Определяем размерность модели
    d_model = d_song + d_album + d_artist

    # Задаем конфигурационный словарь для составных эмбеддингов
    custom_embedding_cfg = {
        "num_songs":    songs_vocab_size,
        "num_albums":   albums_vocab_size,
        "num_artists":  artists_vocab_size,
        "num_features": num_features,
        "d_song":       d_song,
        "d_album":      d_album,
        "d_artist":     d_artist
    }

    # Создаем кастомный эмбеддинг слой
    embedding = CustomEmbiggingLayer(custom_embedding_cfg)

    # Создаем позиционный эмбеддинг слой
    pos_embedding = PositionalEncoding(d_model, seq_len, dropout)

    # Создаем N Блоков кодировщиков
    encoder_blocks = []
    for _ in range(N):
        mh_self_attention = MultiHeadAttentionBlock(d_model, dropout, heads_num)
        pw_feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        encoder_block = EncoderBlock(d_model, mh_self_attention, pw_feed_forward, dropout)
        encoder_blocks.append(encoder_block)

    # Создаем кодировщик
    encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))

    # Создаем слой для проекции
    projection = ProjectionLayer(d_model, songs_vocab_size)

    # Создаем модель
    model = BERT4MusicRecUltimate(encoder, embedding, pos_embedding, projection)

    # Инициализация параметроа модели
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model