FROM python:3.12-slim

# Install system dependencies for Prophet and scientific packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry==latest && \
    poetry config virtualenvs.create false

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry install --no-interaction --no-root && \
    poetry install --no-interaction

# Copy application code
COPY src/ ./src/

# Default command
CMD ["python", "-m", "src.main"]

