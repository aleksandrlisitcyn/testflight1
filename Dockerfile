FROM python:3.10-slim

WORKDIR /app

# Install WeasyPrint dependencies
RUN apt-get update && apt-get install -y \
    libcairo2 \
    libpango-1.0-0 \
    libpangocairo-1.0-0 \
    libgdk-pixbuf-2.0-0 \
    libffi-dev \
    libgl1 \
    libglib2.0-0 \
    shared-mime-info \
    fonts-dejavu \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app
COPY ./assets ./assets

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
