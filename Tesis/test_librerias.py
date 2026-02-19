import cv2
import mediapipe as mp
import psycopg2
from datetime import datetime

# Verificar OpenCV
print("✅ OpenCV versión:", cv2.__version__)

# Verificar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
print("✅ MediaPipe cargado correctamente")

# Verificar conexión a PostgreSQL
try:
    conn = psycopg2.connect(
        dbname="postgres",       # Cambia si usas otra base
        user="postgres",
        password="admin",  # Reemplaza con tu contraseña
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT NOW();")
    fecha = cursor.fetchone()[0]
    print("✅ Conexión a PostgreSQL exitosa. Fecha actual:", fecha)
    conn.close()
except Exception as e:
    print("❌ Error al conectar a PostgreSQL:", e)
