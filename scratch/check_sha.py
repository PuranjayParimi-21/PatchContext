import sqlite3
import os

db_path = 'data/metadata.db'
if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT sha, is_indexed FROM commits WHERE sha LIKE '6a274d1%'")
    rows = cursor.fetchall()
    print("Found rows:", rows)
else:
    print("DB does not exist")
