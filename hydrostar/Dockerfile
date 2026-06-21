FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Set environment variables
ENV PORT=5059
ENV FLASK_APP=app.py

EXPOSE 5059

# Start with Gunicorn, preloading for performance
CMD ["gunicorn", "--bind", "0.0.0.0:5059", "--workers", "2", "--preload", "app:app"]
