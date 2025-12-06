import sqlite3
import os

DB_FILE = "med_dev_qms.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    if not os.path.exists(DB_FILE):
        print("Initializing new database...")
    
    conn = get_db_connection()
    c = conn.cursor()
    
    # QMS Documents Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            version TEXT NOT NULL,
            author TEXT NOT NULL,
            approval_status TEXT NOT NULL,
            last_updated TEXT NOT NULL,
            content TEXT,
            approved_by TEXT,
            approval_date TEXT
        )
    ''')

    # Document Versions Table (One-to-many with documents)
    c.execute('''
        CREATE TABLE IF NOT EXISTS document_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT,
            version TEXT,
            change_description TEXT,
            changed_by TEXT,
            timestamp TEXT,
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    ''')

    # Devices Table
    c.execute('''
        CREATE TABLE IF NOT EXISTS devices (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            description TEXT,
            compliance_standard TEXT
        )
    ''')
    
    # Linking Documents to Clauses (Many-to-Many)
    c.execute('''
        CREATE TABLE IF NOT EXISTS document_clauses (
            doc_id TEXT,
            clause_id TEXT,
            PRIMARY KEY (doc_id, clause_id),
            FOREIGN KEY (doc_id) REFERENCES documents (id)
        )
    ''')

    conn.commit()
    conn.close()
    print("Database initialized.")
