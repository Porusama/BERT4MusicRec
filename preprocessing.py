import os
from dotenv import load_dotenv

import psycopg2 as sql_int
import pandas as pd

from preprocessing_utils import preprocess_data
from config import config, CURRENT_DIR

if __name__ == "__main__":
    load_dotenv()

    # Формируем подключение
    sql_connection = sql_int.connect(
        dbname=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        host=os.getenv("POSTGRES_HOST"),
        password=os.getenv("POSTGRES_PASSWORD"),
        port=os.getenv("POSTGRES_PORT")
    )

    # Производим запрос на данные
    query = open(
        os.path.join(
            CURRENT_DIR, 
            'sql queries\\get_all_described_playlists.sql'
        )
    ).read()

    ram_butcher = pd.read_sql_query(query, con=sql_connection)

    # Предобрабатываем
    preprocess_data(ram_butcher, mode=True)

    ram_butcher.to_csv(
        os.path.join(
            CURRENT_DIR,
            config['dataset_path']
        ), 
        index=False
    )

    sql_connection.close()