from config import config, get_models_weights_file_path
from dataset import PlaylistDataset, dataset_proper_load
from model import build_BERT4MusicRecUltimate_model

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchmetrics.functional.retrieval import retrieval_hit_rate, retrieval_normalized_dcg, retrieval_reciprocal_rank

from pathlib import Path
from tqdm.auto import tqdm
import warnings

from sklearn.model_selection import train_test_split

def get_dataset(config):
    ''' Формирует тренировочный и тестовый загрузчики
    '''
    # Читаем данные
    ds = dataset_proper_load(config['dataset_path'])

    playlist_ids = ds['playlist_id'].unique().to_list()

    ds_train_ids, ds_test_ids = train_test_split(
        playlist_ids,
        train_size=config['train_size']
    )

    # Разделяем датасет на тестовый и 
    ds_train_ids = [int(x[0]) for x in ds_train_ids]
    ds_test_ids  = [int(x[0]) for x in ds_test_ids]

    train_ds_raw = [ds.get_group(x) for x in ds_train_ids]
    test_ds_raw  = [ds.get_group(x) for x in ds_test_ids]

    del ds

    # Получаем датасеты имплементирующиу torch.utils.data.Dataset
    train_ds = PlaylistDataset(config['seq_len'], train_ds_raw, config['cloze_probability'], mode=True)
    test_ds  = PlaylistDataset(config['seq_len'], test_ds_raw,  config['cloze_probability'], mode=False)

    # Получаем загрузчики данных
    train_dataloader = DataLoader(train_ds, config['batch_size'], shuffle=True, pin_memory=True)
    test_dataloader  = DataLoader(test_ds,  config['batch_size'], shuffle=True, pin_memory=True)

    return train_dataloader, test_dataloader

def validate_model(model, validation_ds, device, writer, epoch):
    model.eval()

    hr1_accum = 0
    hr5_accum = 0
    hr10_accum = 0
    ndcg5_accum = 0
    ndcg10_accum = 0
    mrr_accum = 0

    val_global_step = 0

    with torch.no_grad():
        validation_iterator = tqdm(validation_ds, desc=f'Валидируется эпоха: {epoch:02d}')
        for batch, padd_mask, label in validation_iterator:
            # Перемещаем на девайс
            song_ids      = batch['song_ids'].to(device)            # (B, L)
            album_ids     = batch['album_ids'].to(device)           # (B, L)
            artist_ids    = batch['artist_ids'].to(device)          # (B, L)
            num_features  = batch['numeric_features'].to(device)    # (B, L, num_feats)
            padd_mask_gpu = padd_mask.to(device)                    # (B, 1, 1, L)
            label_gpu     = label.to(device)                        # (B, L)

            # Прогоняем тензоры
            encoder_output = model.encode(song_ids, album_ids, artist_ids, num_features, padd_mask_gpu) # (B, L)
            projection_output = model.project(encoder_output)                                           # (B, L, songs_vocab_size)

            # Получаем элементы сессии для которых делаем прогноз
            bool_mask   = (label_gpu != -100)               # bool (B, L)           
            rows, cols = bool_mask.nonzero(as_tuple=True)

            # Получаем прогнозы и таргеты для каждой сесси
            preds = projection_output[rows, cols, :]    # (B, songs_vocab_size)
            target = label_gpu[rows, cols]              # (B)

            # Приводим target в булевый вид для работы с torchmetrics
            target_bool = torch.zeros_like(preds, dtype=torch.bool)
            target_bool[torch.arange(preds.size(0)), target] = True # (B, songs_vocab_size)

            for p, t in zip(preds, target_bool):
                # Обновляем метрики
                hr1_accum    += retrieval_hit_rate(p, t, top_k=1)
                hr5_accum    += retrieval_hit_rate(p, t, top_k=5)
                hr10_accum   += retrieval_hit_rate(p, t, top_k=10)
                ndcg5_accum  += retrieval_normalized_dcg(p, t, top_k=5)
                ndcg10_accum += retrieval_normalized_dcg(p, t, top_k=10)
                mrr_accum    += retrieval_reciprocal_rank(p, t)

            val_global_step += 1

    if writer:
        # Логируем ошибки эпохи 
        writer.add_scalar('validation HR@1', hr1_accum / val_global_step, epoch)
        writer.flush()

        writer.add_scalar('validation HR@5', hr5_accum / val_global_step, epoch)
        writer.flush()

        writer.add_scalar('validation HR@10', hr10_accum / val_global_step, epoch)
        writer.flush()

        writer.add_scalar('validation NDCG@5', ndcg5_accum / val_global_step, epoch)
        writer.flush()

        writer.add_scalar('validation NDCG@10', ndcg10_accum / val_global_step, epoch)
        writer.flush()

        writer.add_scalar('validation MRR', mrr_accum / val_global_step, epoch)
        writer.flush()


def get_model(config):
    model = build_BERT4MusicRecUltimate_model(
        config['unique_songs'],
        config['unique_albums'],
        config['unique_artists'],
        config['numeric_features_amo'],
        config['d_songs'],
        config['d_albums'],
        config['d_artists'],
        config['seq_len'],
        config['d_ff'],
        config['N'],
        heads_num=config['heads'],
        dropout=config['dropout_prob']
    )
    return model


def train_model(config):
    ''' Создает и тренирует модель
    '''
    warnings.filterwarnings('ignore')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'===< Используется устройтсво {device} >===')

    # Создаем папку для сохранения весов модели
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    # Получаем загрузчики
    print('===< Происходит загрузка датасета >===')
    train_dataloader, test_dataloader = get_dataset(config)
    print('===< Загрузка завершена >===')

    model = get_model(config)
    model.to(device)

    # Создаем summary writer для формирования данных для графиков
    writer = SummaryWriter(config['writer_dict'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)

    # Загрузка сохраненного прогресса обучения из файла
    initial_epoch = 1
    global_step = 0
    if config['epoch_to_preload']:
        model_filename = get_models_weights_file_path(config, config['epoch_to_preload'])
        print(f'===< Предзагружается модель {model_filename} >===')
        state = torch.load(model_filename)
        model.load_state_dict(state['model'])
        initial_epoch = state['epoch'] + 1
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']

    # Определяем лосс функцию
    loss_fn = nn.NLLLoss(ignore_index=-100).to(device)

    # Обучаем
    print('===< Начинается обучение >===')
    for epoch in range(initial_epoch, config['num_epochs'] + 1):
        model.train()
        batch_iterator = tqdm(train_dataloader, desc=f'Обрабатывается эпоха: {epoch:02d}')
        for batch, padd_mask, label in batch_iterator:
            # Перемещаем на девайс
            song_ids      = batch['song_ids'].to(device)            # (B, L)
            album_ids     = batch['album_ids'].to(device)           # (B, L)
            artist_ids    = batch['artist_ids'].to(device)          # (B, L)
            num_features  = batch['numeric_features'].to(device)    # (B, L, num_feats)
            padd_mask_gpu = padd_mask.to(device)                    # (B, 1, 1, L)
            label_gpu     = label.to(device)                        # (B, L)

            # Прогоняем тензоры
            encoder_output = model.encode(song_ids, album_ids, artist_ids, num_features, padd_mask_gpu) # (B, L)
            projection_output = model.project(encoder_output)                                           # (B, L, songs_vocab_size)

            # Вычисляем ошибку
            # (B, L, songs_vocab_size) --(.view)--> (B*L, songs_vocab_size)
            loss = loss_fn(torch.log(projection_output.view(-1, config['unique_songs']) + 1e-9), label_gpu.view(-1))

            # Красивый вывод ошибки
            batch_iterator.set_postfix({"Ошибка": f"{loss.item():6.3f}"})

            # Логируем ошибку
            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            # Backprop
            loss.backward()

            # Обновляем веса
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Валидируем модель каждую эпоху
        validate_model(model, test_dataloader, device, writer, epoch)

        # Сохраняем модель и оптимизатор каждую эпоху
        model_filename = get_models_weights_file_path(config, epoch)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, model_filename)

    print('===< Модель проучена >===')

if __name__ == "__main__":
    print('===< BERT4MusicRec.ULTIMATE >===')
    train_model(config)