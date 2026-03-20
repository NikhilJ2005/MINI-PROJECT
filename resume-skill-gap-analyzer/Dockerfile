# Stage 1: Build React frontend
FROM node:20-alpine AS frontend-build
WORKDIR /app
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ .
# Ensure production build uses empty base URL (same-origin API calls)
ENV VITE_API_BASE_URL=""
RUN npm run build

# Stage 2: Python backend + serve frontend static files
FROM python:3.11-slim AS backend

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy backend code
COPY backend/ .

# Build training datasets at build time (uses streaming HF data + simulation)
RUN python build_datasets.py || echo "Dataset build completed with warnings (non-fatal)"

# Copy built frontend into static directory
COPY --from=frontend-build /app/dist ./static

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
