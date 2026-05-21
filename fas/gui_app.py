import tkinter as tk
from tkinter import messagebox, simpledialog
import pandas as pd
import sqlite3

DB_PATH = 'fas_app.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_table():
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS patients (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id TEXT,
        preop_score REAL,
        postop_score REAL,
        tms_sessions INTEGER
    )''')
    conn.commit()
    conn.close()

class FASApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FAS TMS Quantum ML Desktop App")
        self.geometry("500x400")
        ensure_table()
        self.create_widgets()

    def create_widgets(self):
        tk.Button(self, text="Add Patient Data", command=self.add_patient).pack(fill='x')
        tk.Button(self, text="Update TMS Sessions", command=self.update_tms_sessions).pack(fill='x')
        tk.Button(self, text="Analyze Patient", command=self.analyze_patient).pack(fill='x')
        tk.Button(self, text="View Database", command=self.view_database).pack(fill='x')
        tk.Button(self, text="Exit", command=self.destroy).pack(fill='x')
        self.output = tk.Text(self, height=15)
        self.output.pack(fill='both', expand=True)

    def add_patient(self):
        pid = simpledialog.askstring("Patient ID", "Enter Patient ID:")
        pre = simpledialog.askfloat("Pre-op Score", "Enter Pre-op Score (0-100):")
        post = simpledialog.askfloat("Post-op Score", "Enter Post-op Score (0-100):")
        tms = simpledialog.askinteger("TMS Sessions", "Enter Number of TMS Sessions:")
        if None in (pid, pre, post, tms):
            return
        conn = get_db_connection()
        conn.execute("INSERT INTO patients (patient_id, preop_score, postop_score, tms_sessions) VALUES (?, ?, ?, ?)", (pid, pre, post, tms))
        conn.commit()
        conn.close()
        messagebox.showinfo("Saved", f"Data saved for Patient {pid}")

    def update_tms_sessions(self):
        pid = simpledialog.askstring("Patient ID", "Enter Patient ID to update:")
        tms = simpledialog.askinteger("TMS Sessions", "Enter new TMS session count:")
        if None in (pid, tms):
            return
        conn = get_db_connection()
        conn.execute("UPDATE patients SET tms_sessions = ? WHERE patient_id = ?", (tms, pid))
        conn.commit()
        conn.close()
        messagebox.showinfo("Updated", "TMS sessions updated.")

    def analyze_patient(self):
        pid = simpledialog.askstring("Patient ID", "Enter Patient ID to analyze:")
        if not pid:
            return
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM patients", conn)
        conn.close()
        row = df[df['patient_id'] == pid]
        if row.empty:
            messagebox.showerror("Error", "Patient not found.")
            return
        preop_score = row.iloc[0]['preop_score']
        postop_score = row.iloc[0]['postop_score']
        tms_sessions = row.iloc[0]['tms_sessions']
        pre_q = preop_score / 100
        post_q = postop_score / 100
        self.output.delete('1.0', tk.END)
        self.output.insert(tk.END, f"Quantum Pre-op Score: {pre_q:.3f}\n")
        self.output.insert(tk.END, f"Quantum Post-op Score: {post_q:.3f}\n")
        self.output.insert(tk.END, f"TMS Sessions: {tms_sessions}\n")
        self.output.insert(tk.END, f"Recovery Progress: Pre-op {preop_score} -> Post-op {postop_score}\n")

    def view_database(self):
        conn = get_db_connection()
        df = pd.read_sql_query("SELECT * FROM patients", conn)
        conn.close()
        self.output.delete('1.0', tk.END)
        if df.empty:
            self.output.insert(tk.END, "No patient data available.\n")
        else:
            self.output.insert(tk.END, df.to_string(index=False))

if __name__ == '__main__':
    app = FASApp()
    app.mainloop()
