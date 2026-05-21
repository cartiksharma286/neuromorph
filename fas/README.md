# FAS Treatment App with TMS and Quantum Machine Learning

This is a Streamlit-based web app for managing FAS (Foreign Accent Syndrome) treatment using Transcranial Magnetic Stimulation (TMS) and analyzing pre-op and post-op recovery scores with quantum machine learning.

## Features
- Patient data entry (pre-op and post-op scores)
- TMS session tracking
- Quantum machine learning analysis (Pennylane/Qiskit)
- Visualization of recovery progress
- Database-backed patient/session storage
- Improved UI with navigation sidebar
- TMS session tracking and update
- Quantum ML analysis and recovery visualization per patient
- View all patient data in-app

## Setup
1. Install dependencies:
   ```bash
   pip install streamlit pennylane qiskit matplotlib pandas
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```

## Database
- Uses SQLite (fas_app.db) for persistent storage
- See db.py for schema and initialization

## Usage
- Enter patient and session data
- View quantum ML-based analysis and recovery visualization

## Advanced Usage
- Extend quantum_score_analysis in app.py for more advanced analytics
- Add clinical scoring logic as needed

---
*This is a prototype. Replace placeholder logic with clinical algorithms as needed.*
