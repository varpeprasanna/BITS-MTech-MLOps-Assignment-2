FROM python:3.11-slim

# Prevent Python from creating .pyc files
# and make stdout/stderr immediately visible.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install serving dependencies first.
# This improves Docker layer caching.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the inference service needs.
COPY api ./api
COPY src ./src
COPY artifacts/best_model.pt ./artifacts/best_model.pt

# FastAPI service port
EXPOSE 8000

CMD ["uvicorn",\
    "api.main:app",\
    "--host",\
    "0.0.0.0",\
    "--port",\
    "8000"]