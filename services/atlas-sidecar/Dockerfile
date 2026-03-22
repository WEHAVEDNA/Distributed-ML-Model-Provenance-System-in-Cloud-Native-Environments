# ── Stage 1: Build atlas-cli from source ─────────────────────────────────────
# Atlas CLI has no published Docker image, so we compile from the official repo.
FROM rust:1.88-slim AS atlas-builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# cargo install pulls from crates.io (mirrors the IntelLabs/atlas-cli release).
# Pin to a specific version for reproducibility.
RUN cargo install atlas-cli --root /atlas-install 2>&1 | tail -5

# ── Stage 2: Python FastAPI sidecar ──────────────────────────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    openssl \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled atlas-cli binary from the builder stage
COPY --from=atlas-builder /atlas-install/bin/atlas-cli /usr/local/bin/atlas-cli

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .

# Atlas manifests and signing key are stored on this volume.
# Mount a PVC here in Kubernetes; a named volume in docker-compose.
VOLUME /manifests

ENV MANIFESTS_DIR=/manifests/atlas

EXPOSE 8004
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8004"]
