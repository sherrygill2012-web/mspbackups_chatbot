# Use slim Python image
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install CPU-only PyTorch first (much smaller than GPU version)
# Then install other requirements
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    # Clean up pip cache
    rm -rf /root/.cache/pip

# Copy application code
COPY *.py ./
COPY pages/ pages/
COPY .streamlit/ .streamlit/

# Create directories for persistence
RUN mkdir -p /app/data

# Expose ports (Streamlit and API)
EXPOSE 8501 8000

# Health check for Streamlit
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Default to Streamlit, can be overridden
# To run API: docker run -e RUN_MODE=api ...
# To run Slack bot: docker run -e RUN_MODE=slack ...
ENV RUN_MODE=streamlit

# Entrypoint script
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

ENTRYPOINT ["/docker-entrypoint.sh"]
