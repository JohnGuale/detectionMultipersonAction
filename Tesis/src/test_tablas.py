from Resources.Conexion import get_connection

def listar_tablas():
    conn = get_connection()
    if conn:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
        """)
        tablas = cursor.fetchall()
        print("✅ Tablas disponibles en la base de datos:")
        for tabla in tablas:
            print("-", tabla[0])
        cursor.close()
        conn.close()
    else:
        print("❌ No se pudo conectar a la base de datos.")

if __name__ == "__main__":
    listar_tablas()
