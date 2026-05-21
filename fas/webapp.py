from flask import Flask, render_template_string, request, redirect, url_for
import pandas as pd
import sqlite3
import os

DB_PATH = 'fas_app.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

app = Flask(__name__)

HOME_HTML = '''
<h2>FAS & FASD DBS Web App</h2>
<ul>
  <li><a href="/fas">Foreign Accent Syndrome (FAS) TMS</a></li>
  <li><a href="/fasd">Fetal Alcohol Syndrome (FASD) DBS</a></li>
</ul>
'''

FAS_FORM = '''
<h3>Add FAS Patient</h3>
<form method="post">
  Patient ID: <input name="patient_id"><br>
  Pre-op Score: <input name="preop_score" type="number" min="0" max="100"><br>
  Post-op Score: <input name="postop_score" type="number" min="0" max="100"><br>
  TMS Sessions: <input name="tms_sessions" type="number"><br>
  <input type="submit" value="Add">
</form>
<a href="/fas">Back</a>
'''

FASD_FORM = '''
<h3>Add FASD Patient</h3>
<form method="post">
  Patient ID: <input name="patient_id"><br>
  Pre-op Score: <input name="preop_score" type="number" min="0" max="100"><br>
  Post-op Score: <input name="postop_score" type="number" min="0" max="100"><br>
  DBS Sessions: <input name="dbs_sessions" type="number"><br>
  <input type="submit" value="Add">
</form>
<a href="/fasd">Back</a>
'''

@app.route('/')
def home():
    return render_template_string(HOME_HTML)

@app.route('/fas', methods=['GET', 'POST'])
def fas():
    conn = get_db_connection()
    if request.method == 'POST':
        conn.execute("INSERT INTO patients (patient_id, preop_score, postop_score, tms_sessions) VALUES (?, ?, ?, ?)",
                     (request.form['patient_id'], request.form['preop_score'], request.form['postop_score'], request.form['tms_sessions']))
        conn.commit()
        return redirect(url_for('fas'))
    df = pd.read_sql_query("SELECT * FROM patients", conn)
    conn.close()
    table = df.to_html(index=False) if not df.empty else '<i>No data</i>'
    return render_template_string(FAS_FORM + '<h4>Database</h4>' + table)

@app.route('/fasd', methods=['GET', 'POST'])
def fasd():
    conn = get_db_connection()
    conn.execute("CREATE TABLE IF NOT EXISTS fasd_patients (id INTEGER PRIMARY KEY AUTOINCREMENT, patient_id TEXT, preop_score REAL, postop_score REAL, dbs_sessions INTEGER)")
    if request.method == 'POST':
        conn.execute("INSERT INTO fasd_patients (patient_id, preop_score, postop_score, dbs_sessions) VALUES (?, ?, ?, ?)",
                     (request.form['patient_id'], request.form['preop_score'], request.form['postop_score'], request.form['dbs_sessions']))
        conn.commit()
        return redirect(url_for('fasd'))
    df = pd.read_sql_query("SELECT * FROM fasd_patients", conn)
    conn.close()
    table = df.to_html(index=False) if not df.empty else '<i>No data</i>'
    return render_template_string(FASD_FORM + '<h4>Database</h4>' + table)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
