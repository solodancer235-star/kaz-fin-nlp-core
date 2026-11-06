# syntax=docker/dockerfile:1
# ---------- Stage 1: build wheels (keeps compilers out of the runtime image) ----------
FROM python:3.12-slim AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# ---------- Stage 2: minimal runtime ----------
FROM python:3.12-slim AS runtime
ARG GIT_COMMIT=unknown
ARG BUILD_NUMBER=local
LABEL org.opencontainers.image.title="kaz-fin-nlp-core" \
      org.opencontainers.image.description="Bilingual KZ/RU financial news NLP API" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.version="0.2.0-b${BUILD_NUMBER}"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels \
 && useradd --create-home --uid 10001 appuser

COPY kaz_fin_nlp/ ./kaz_fin_nlp/
USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health').status==200 else 1)"

CMD ["sh", "-c", "uvicorn kaz_fin_nlp.api:app --host 0.0.0.0 --port ${PORT}"]
