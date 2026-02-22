import psycopg2
import os
from dotenv import load_dotenv
load_dotenv()

# Configuración de la conexión
host = 'localhost'
port = '5432'
database = 'multipleestimationdb'
user = 'postgres'
password = 'admin'
encoding = 'utf8'

def get_connection():
    """Establece y devuelve una conexión a la base de datos PostgreSQL."""
    try:
        connection = psycopg2.connect(
            host=host,
            port=port,
            database=database,
            user=user,
            password=password,
            client_encoding=encoding
        )
        print("Conexión exitosa")
        return connection
    except psycopg2.Error as e:
        print(f"Error en la conexión: {e.pgcode} - {e.pgerror}")
        return None