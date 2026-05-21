import sqlite3
import os

def get_db_connection():
    conn = sqlite3.connect('fas_app.db')
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists('fas_app.db'):
        conn = get_db_connection()
        conn.execute('''
            CREATE TABLE patients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT,
                preop_score REAL,
                postop_score REAL,
                tms_sessions INTEGER
            )
        ''')
        conn.commit()
        conn.close()

init_db()
