# ======================================================================
# STAGE 1: Dependency Compiler Engine (Build Environment)
# ======================================================================
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt


# ======================================================================
# STAGE 2: Secure Hardened Runtime Image (Final Production Target)
# ======================================================================
FROM python:3.11-slim AS production

WORKDIR /app

# Non-root security user setup
RUN groupadd -r raguser && useradd -r -g raguser raguser

COPY --from=builder /opt/venv /opt/venv
COPY . .

ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

RUN chown -R raguser:raguser /app /opt/venv
USER raguser

EXPOSE 8000

# Docker level Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Dynamic port binding for Render compatibility
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]