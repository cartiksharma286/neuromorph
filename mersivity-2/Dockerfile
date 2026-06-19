# Use slim Python 3.9 base (matches local dev environment)
FROM python:3.9-slim

# Install system dependencies needed by scikit-image, scipy, and trimesh
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app.py registration_utils.py snr_optimizer.py ./
COPY generate_nature_preprint.py generate_nature_eeg_report.py ./
COPY generate_nature_eeg_report_plots.py generate_nature_preprint_plots.py ./
COPY templates/ templates/

# Copy data files (DICOM + STL meshes)
COPY mri/ mri/
COPY IMAGES/ IMAGES/
COPY MendMesh_Triplane_Setup_Ness_02.stl .

# Cloud Run sets PORT env var; default to 8080
ENV PORT=8080

# Use gunicorn for production (4 workers, 120s timeout for heavy registration compute)
CMD exec gunicorn --bind :$PORT --workers 2 --threads 4 --timeout 120 --preload app:app
